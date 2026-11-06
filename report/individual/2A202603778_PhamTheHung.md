# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603778
- **Họ và tên:** Phạm Thế Hùng
- **SEED** (in ở ô đầu notebook): 6254

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ Có — đã kiểm tra |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ Có — đã kiểm tra |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ Có — đã kiểm tra |

### Kiểm tra bằng chứng bổ sung

| Tệp | Kết quả kiểm tra |
| --- | --- |
| `day01_lab.ipynb` | Có; chứa code và output, in `SEED=6254`, đủ 10 khóa dự đoán, báo cáo cuối ghi 15/15 kiểm tra đạt. |
| `lab1_report.json` | Có; chứa `seed`, `predictions`, `checks`, `experiments`, `challenge`, `results`; cả 15 giá trị kiểm tra đều là `true`. |
| `hand_calc.jpg` | Có; đã xem ảnh viết tay, thấy các bước tính baseline, sigmoid, F1, cosine và chi phí LLM. |

Dữ liệu sau bỏ trùng gồm 5.169 tin: 4.516 ham và 653 spam. Tập train có 3.619 tin, validation 775 tin, test 775 tin; test có 88 spam. SUB gồm 40 tin lấy từ test, cân bằng 20 spam và 20 ham. Vocabulary BoW có 2.000 từ và encoder tạo embedding 384 chiều.

Đối chiếu phép tính tay: baseline = 4.516/5.169 ≈ 0,87; sigmoid(0) = 0,50; F1 = 160/(160+20+40) ≈ 0,73; cosine = 4/(√5 × 3) ≈ 0,60. Với giả định bài tập 1 triệu tin/ngày, 100 token input và 1 token output/tin, chi phí = 100 × 0,10 + 1 × 0,50 = 10,50 USD/ngày. Đây là phép tính theo giá giả định trong đề, không phải xác nhận giá API hiện hành.

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

### Bảng kết quả bổ sung từ `lab1_report.json`

| Phương án | F1 test (775 tin) | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 ms/tin | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.9268 | 0.9744 | 6.635 | 382.905 | 10.187 | 0.00000000 |
| 2. BoW + MLP (deep learning) | 0.9461 | 1.0000 | 14.023 | 644.995 | 3.901 | 0.00000000 |
| 3a. Encoder (MiniLM) + kNN | 0.8718 | 0.9744 | 79.763 | 42.131 | 58.693 | 0.00000000 |
| 3b. Encoder (MiniLM) + LogReg | 0.9364 | 0.9474 | 80.254 | 49.393 | 38.040 | 0.00000000 |
| 3c. Encoder (MiniLM) + MLP head | 0.9419 | 0.9744 | 80.817 | 45.534 | 59.684 | 0.00000000 |
| 4a. LLM nhỏ zero-shot | Chưa đo toàn test | 0.0000 | 0.000 | 1.142 | 1425.215 | 0.00680333 |
| 4b. LLM lớn zero-shot | Chưa đo toàn test | 0.0000 | 0.000 | 1.135 | 1211.301 | 0.06803333 |
| 4c. LLM nhỏ few-shot | Chưa đo toàn test | 0.0000 | 0.000 | 1.084 | 1764.887 | 0.01827000 |

Các hàng LLM trong JSON có trường `f1`, nhưng code chỉ đánh giá trên SUB 40 rồi lưu cùng giá trị vào `f1_SUB`; không có bằng chứng LLM đã chạy trên toàn bộ 775 tin test. So sánh chất lượng giữa mọi phương án phải dùng cột SUB. SUB nhỏ nên không nên xếp hạng theo chênh lệch rất nhỏ.

BoW + MLP đạt F1 test 0,9461, precision 1,0000, recall 0,8977; tương ứng phát hiện 79/88 spam, bỏ sót 9 spam và không chặn nhầm ham trên tập test này. BoW + LogReg có F1 0,9268, precision 1,0000 và recall 0,8636. MiniLM + LogReg đạt F1 0,9364, precision 0,9529 và recall 0,9205: bắt được nhiều spam hơn BoW + MLP nhưng có đánh đổi về chặn nhầm và độ trễ.

**Giới hạn quan trọng ở LLM:** output các ô 72 và 75 đều ghi `invalid=40`. Hàm `llm_classify_batch` đổi nhãn không đọc được (`None`) thành ham (0), nên trên SUB cân bằng, accuracy = 0,5 và F1 spam = 0. Kết quả này phản ánh pipeline chưa tạo/đọc được nhãn hợp lệ; chưa đủ cơ sở kết luận năng lực phân loại thực của LLM thấp hơn ML. Kiểm tra mock đạt không bảo đảm phản hồi API thật hợp lệ. Notebook không lưu phản hồi thô của các lượt lỗi nên chưa xác định được nguyên nhân gốc.

Chi phí LLM trong bảng được ước tính bằng số từ/0,75 và bảng giá trong notebook, không phải hóa đơn hoặc token usage thực. Chi phí 0 của mô hình cục bộ là không tính phí API; vẫn có chi phí máy chủ, điện và vận hành. Độ trễ tính từ văn bản thô đến nhãn trong môi trường chạy lab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

### Ba dự đoán lệch rõ so với output

1. **`ml_f1`, 13:03:58:** dự đoán F1 khoảng 0,85, precision cao hơn recall. Thực tế BoW + LogReg đạt F1 0,9268, cao hơn 0,0768; precision 1,0000 > recall 0,8636 đúng với dự đoán định tính. Giải thích hợp lý là vocabulary BoW nắm được tín hiệu spam khá tốt trên phân phối test; tuy nhiên, các mẫu spam chỉ khác mã hoặc số vẫn có thể giống train dù đã bỏ trùng tuyệt đối, khiến test lạc quan hơn stress test.
2. **`llm_models`, 13:08:51:** dự đoán model lớn có F1 cao hơn model nhỏ khoảng 0,02–0,05. Thực tế cả hai cùng F1 SUB = 0,0000, chênh lệch bằng 0; chi phí model lớn đúng là gấp 10 lần model nhỏ trong lần đo này. Nguyên nhân trực tiếp của kết quả nhãn là cả hai đều có 40 phản hồi không hợp lệ và bị fallback sang ham. Do đó sai lệch này cho thấy cần kiểm tra pipeline trước khi so sánh model, chưa chứng minh model lớn không có lợi thế về năng lực.
3. **`llm_fewshot`, 13:10:05:** dự đoán few-shot tăng F1 nhẹ và chi phí tăng 2–3 lần. Thực tế F1 vẫn bằng 0,0000; chi phí tăng từ 0,00680333 lên 0,01827000 USD/1.000 tin, khoảng 2,69 lần, còn p95 tăng từ 1.425,215 lên 1.764,887 ms. Dự đoán chi phí phù hợp nhưng chất lượng không tăng vì few-shot vẫn có `invalid=40`; thêm ví dụ chưa khắc phục được phản hồi không hợp lệ.

Đây là ba trường hợp có bằng chứng rõ, không phải xếp hạng định lượng toàn bộ 10 dự đoán vì nhiều dự đoán chỉ mang tính định tính. Riêng `dl_relu` dự đoán F1 giảm nhẹ hoặc gần như không đổi: output bỏ ReLU đạt 0,952, cao hơn mức 0,946 của MLP gốc khoảng 0,006. Bỏ ReLU khiến hai lớp affine gộp thành một phép affine; mô hình có biên quyết định tuyến tính, nhưng quá trình tối ưu và tham số hóa vẫn khác LogReg tự code.

Các dự đoán và thí nghiệm đầu tiên được ghi cùng giây trong log, nên dấu thời gian chỉ có độ phân giải giây chưa tự chứng minh khoảng cách thời gian; thứ tự cell cho thấy lệnh `predict(...)` đặt trước `require_prediction(...)`.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

### Kết quả stress test và giới hạn số lượng case

Bộ CSV gốc có đúng 50 case (21 ham, 29 spam). Notebook bổ sung 5 `MY_CASES` (3 ham, 2 spam). Output ô 87 có accuracy 0,509; 0,527; 0,655; 0,636 và 0,436, tương ứng 28/55, 29/55, 36/55, 35/55 và 24/55 khi làm tròn. Bảng theo nhóm cũng có 92% trên nhóm ham giống spam và 9% trên nhóm spam tự nhiên, phù hợp mẫu số đã tăng thành 12 và 11. Vì vậy bảng đã lưu phù hợp với **55 case sau bổ sung**, dù tiêu đề code vẫn ghi “50 case”. Không gán các số này thành kết quả riêng của 50 case gốc khi chưa tách lại bằng dự đoán từng tin.

| Model | F1 SUB thường | F1 bộ khó đã lưu | Mức giảm F1 | Accuracy bộ khó đã lưu |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0,974 | 0,308 | 0,666 | 0,509 |
| BoW + MLP | 1,000 | 0,381 | 0,619 | 0,527 |
| Embedding + kNN | 0,974 | 0,689 | 0,285 | 0,655 |
| Embedding + LogReg | 0,947 | 0,615 | 0,332 | 0,636 |
| Embedding + MLP head | 0,974 | 0,615 | 0,359 | 0,636 |
| LLM nhỏ zero-shot | 0,000 | 0,000 | 0,000 | 0,436 |
| LLM nhỏ + prompt phòng thủ | 0,000* | 0,000 | Không có baseline riêng | 0,436 |

Số liệu trong bảng làm tròn theo output ô 87. (*) Cột SUB của prompt phòng thủ được code lấy từ hàng zero-shot 4a, không phải một phép đo SUB riêng của 4d.

**BoW + LogReg tụt nhiều nhất theo mốc SUB**: khoảng 0,666 F1. Nếu lấy F1 toàn test làm mốc cho các mô hình cục bộ, nó vẫn giảm lớn nhất: 0,9268 − 0,308 ≈ 0,619. LLM có F1 thấp nhưng không phải model giảm nhiều nhất vì baseline đã bằng 0.

BoW + LogReg chỉ đúng khoảng 9% nhóm spam giọng tự nhiên, 12% nhóm thao túng và 14% nhóm né từ khóa. BoW + MLP đúng 0% nhóm né từ khóa và khoảng 9% spam giọng tự nhiên. Encoder + kNN bắt tốt nhóm thao túng (100%) nhưng đúng chỉ 20% nhóm nói về spam; việc có ngữ nghĩa gần spam không đồng nghĩa người gửi đang thực hiện hành vi spam. Các tỷ lệ theo nhóm là accuracy, khác với F1 spam ở bảng tổng hợp.

Hai case cụ thể có bằng chứng mọi model đều sai ở ô 88:

- **Case #12**, nhãn thật spam: lời mở đầu lịch sự kiểu “hi there, sorry to bother you”, giả vờ được bạn giới thiệu rồi dẫn tới trang hẹn hò. Vì đây là một case nhị phân mà mọi model đều sai, nhãn dự đoán là ham. Cơ chế khả dĩ: cách diễn đạt giống hội thoại bình thường, thiếu các token spam nổi bật; embedding cũng có thể gần tin làm quen hợp lệ. Đây là giải thích theo cấu trúc đầu vào, chưa phải kết quả đo đóng góp từng đặc trưng.
- **Case #31**, nhãn thật spam: chèn dấu chấm thành `s.e.l.e.c.t.e.d`, `c.a.s.h`, `r.e.w.a.r.d`, `c.a.l.l`. Mọi model dự đoán ham. Tokenizer BoW tách thành các mảnh ngắn nên mất từ khóa vốn có; encoder cũng có thể chịu ảnh hưởng do văn bản bị biến dạng. Có thể thử chuẩn hóa có kiểm soát hoặc đặc trưng ký tự, rồi đo lại false positive để tránh ghép nhầm chữ hoặc làm hỏng mã giao dịch.

Ở nhóm injection, 4a và 4d cùng đúng 17% (1/6); chưa thấy lợi ích của prompt phòng thủ trong output đã lưu. Không nên quy mọi lỗi này cho injection khi pipeline LLM trên SUB vốn đã có vấn đề nhãn. Có thể dùng kiểm tra định dạng nhãn độc lập, tách dữ liệu SMS khỏi chỉ dẫn và dùng bộ phân loại khác kiểm tra chéo; phản hồi không hợp lệ cần có trạng thái lỗi riêng thay vì mặc nhiên coi là ham.

Bộ khó được tạo có chủ đích để tìm điểm yếu, có phân phối khác train/test. Nó hữu ích cho kiểm thử hồi quy và xây dựng dữ liệu bổ sung; không đại diện tần suất spam ngoài đời và không dùng trực tiếp để dự báo chất lượng vận hành. Chưa có dự đoán từng tin được lưu đủ để tách toàn bộ kết quả 50 case gốc khỏi 5 case tự thêm.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

### Đề được giao và kết quả đã chạy

`SEED % 6 = 6254 % 6 = 2`: chỉ số 2 trong danh sách Python, tức **đề #3** — đổi lớp ẩn MLP thành 4 và 512 nơ-ron, so sánh tham số, F1 test và thời gian train.

Ô 96 đã có code đặt lại `torch.manual_seed(SEED)` cho mỗi cấu hình, gọi `build_mlp(X_tr.shape[1], hidden)`, đếm tham số, huấn luyện 30 epoch trên cùng train/validation rồi đánh giá test. Báo cáo chỉ tổng hợp kết quả có sẵn, không bổ sung code làm thử thách.

| Số nơ-ron lớp ẩn | Số tham số | F1 test (output làm tròn) | Thời gian train |
| ---: | ---: | ---: | ---: |
| 4 | 8.009 | 0,952 | 4,03 s |
| 512 | 1.025.025 | 0,940 | 28,97 s |

Với 2.000 đầu vào và 1 đầu ra, số tham số là `2000h + h + h + 1 = 2002h + 1`. Thay h = 4 được 8.009; h = 512 được 1.025.025, khoảng 128 lần nhiều hơn. Mạng 512 train chậm hơn khoảng 7,19 lần nhưng F1 thấp hơn khoảng 0,012 trong lần chạy này.

Mạng lớn không bảo đảm tổng quát hóa tốt hơn. Dữ liệu hữu hạn và đặc trưng BoW có thể khiến dung lượng thêm không giúp, hoặc tăng nguy cơ overfitting; riêng hai số F1 chưa đủ chứng minh overfitting nếu thiếu đối chiếu đường train/validation cho từng cấu hình. Với kết quả đang có, mạng nhỏ hợp lý khi cần giảm tài nguyên và thời gian huấn luyện. Cần dùng validation và nhiều lần thử để chốt cấu hình, tránh chọn mạng chỉ vì F1 test cao nhất. Không có benchmark p95 riêng cho hai mạng thử thách nên chưa kết luận mạng 4 đạt độ trễ cụ thể.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

### Lựa chọn và đánh đổi theo số liệu lab

Chọn **BoW + MLP 64 nơ-ron của bảng so sánh** làm phương án ban đầu: F1 test 0,9461; F1 SUB 1,0000; train 14,023 s; throughput khoảng 645 tin/s; p95 3,901 ms/tin. Các phương án còn đạt p95 < 50 ms trong lab là BoW + LogReg (10,187 ms) và MiniLM + LogReg (38,040 ms). MiniLM + kNN (58,693 ms), MiniLM + MLP head (59,684 ms) và các LLM đều vượt giới hạn.

10 triệu tin/ngày tương đương khoảng 115,74 tin/s trung bình; throughput cả lô của BoW + MLP cao hơn khoảng 5,57 lần mức trung bình này. Đây là chỉ báo công suất ban đầu, không phải cam kết SLA dưới tải thực: p95 gọi từng tin và throughput cả lô là hai phép đo khác nhau, còn thời gian chờ hàng đợi, mạng, tải đỉnh và tài nguyên triển khai chưa được đo. Cần kiểm thử tải trên hệ thống đích trước khi chốt số máy.

Đánh đổi chính là bỏ sót spam mới hoặc né từ khóa: recall test 0,8977 và F1 bộ khó chỉ 0,381. Precision 1,0 trên một tập test không bảo đảm không chặn nhầm trong sản phẩm. Cần theo dõi false positive, drift, tỷ lệ bỏ sót, bổ sung mẫu mới và chọn ngưỡng bằng validation theo chi phí chặn nhầm tin hợp lệ.

**Thiết kế kết hợp:** BoW + MLP xử lý phần lớn tin. Tin có xác suất ở vùng lưỡng lự, ví dụ [0,3; 0,7], có thể chuyển MiniLM + LogReg để kiểm tra thêm. Xác suất cần được kiểm tra hiệu chỉnh và ngưỡng phải xác lập trên validation. Tổng độ trễ các chặng, hàng đợi và tỷ lệ đi nhánh phụ phải được đo; cộng các p95 riêng không tạo thành p95 của hệ thống. Với ràng buộc 50 ms, LLM phù hợp hơn cho phân tích bất đồng bộ/kiểm tra mẫu và hỗ trợ tạo dữ liệu để con người duyệt. Không đưa API có p95 hơn 1 giây vào đường xử lý bắt buộc của mọi SMS.

Nếu dùng LLM cho toàn bộ 300 triệu tin trong tháng 30 ngày, ngoại suy từ chi phí ước tính của lần chạy hiện có:

| Cấu hình LLM | USD/tháng ước tính |
| --- | ---: |
| Nhỏ zero-shot | 2.041,00 |
| Lớn zero-shot | 20.410,00 |
| Nhỏ few-shot | 5.481,00 |

Công thức: `300.000.000 / 1.000 × usd_per_1k`. Nếu chỉ xử lý tỷ lệ r số tin, phần chi phí API tương ứng xấp xỉ r lần mức trên, với giả định độ dài prompt/response giữ nguyên. Các ước tính dùng cấu hình và bảng giá đã lưu, chưa bao gồm vận hành, retry, thay đổi độ dài tin; pipeline LLM hiện tại cần sửa lỗi nhãn trước khi đánh giá hiệu quả chi phí.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
### AI log bổ sung và phản ánh

AI log dưới đây phân biệt nội dung được ghi sẵn trong notebook với hỗ trợ hoàn thiện báo cáo lần này. Không tự xác nhận rằng người học đã thực hiện thao tác kiểm chứng chưa có bằng chứng.

| Nguồn / phần | Nội dung hỏi AI | Hỗ trợ được ghi nhận | Sửa / kiểm chứng được ghi nhận |
| --- | --- | --- | --- |
| Log sẵn trong notebook, Phần 1 | BoW, Logistic Regression, F1, leakage | Giải thích khái niệm, công thức | Notebook ghi đã đối chiếu bài học. |
| Log sẵn trong notebook, Phần 3 | Embedding phân biệt spam/ham; so LogReg với MLP | Giải thích vector nghĩa và F1 | Notebook ghi xem biểu đồ, so kết quả 3b và 3c. |
| Log sẵn trong notebook, Phần 4 | Gọi API, prompt zero/few-shot, parser, phân loại hàng loạt và chi phí | Hỗ trợ code, phân tích lỗi và chỉ số | Notebook ghi sửa import, chạy mock và đối chiếu output; phản hồi thực trên SUB vẫn invalid 40/40. |
| Lần bổ sung báo cáo này | Tìm, đọc toàn bộ file Markdown và hoàn thiện bằng cách chỉ bổ sung | AI tổng hợp JSON, output notebook, ảnh tính tay; thêm bảng kết quả và phân tích vào các mục 1–7 | AI đối chiếu SEED, 15 checks, phạm vi SUB, fallback nhãn, số lượng case và kết quả thử thách; kiểm tra thay đổi chỉ có phần chèn. Người học cần tự đọc và xác nhận các diễn giải trước khi nộp. |

Phản ánh dựa trên bằng chứng của bài: accuracy cao không đủ trên dữ liệu lệch lớp, và F1 test tốt vẫn có thể giảm mạnh khi gặp spam khác phân phối. Embedding giúp một số tin ngụy trang nhưng vẫn có thể nhầm nội dung nói về spam với hành vi spam. Với LLM, phải kiểm tra tỷ lệ nhãn hợp lệ và cơ chế fallback trước khi suy luận về chất lượng; mô hình phù hợp còn phụ thuộc độ trễ, công suất và chi phí.

### Những điểm bằng chứng chưa giải quyết được

- Chưa có phản hồi LLM thô để xác định nguyên nhân `invalid=40`; không dựng lại hoặc suy đoán như một kết luận đã kiểm chứng.
- Chưa có bảng riêng cho đúng 50 case gốc sau khi loại 5 case bổ sung; giữ số đã lưu và chú thích phạm vi.
- Thử thách chỉ lưu F1 đến 3 chữ số và thời gian đến 2 chữ số; không bổ sung độ chính xác giả.
- Các nội dung nhận xét do AI bổ sung đã được khai báo ở trên; không thay thế bằng chứng người học tự giải thích hoặc kiểm chứng bài.
