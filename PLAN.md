# Kế hoạch so sánh CARS và MDM-VGB trên QM9

## 1. Mục tiêu và phạm vi

Xây dựng thí nghiệm trả lời hai câu hỏi:

1. Khi dùng chung mô hình QM9, reward và ngân sách tính toán, CARS và MDM-VGB khác nhau thế nào về tỷ lệ thành công, tính đa dạng và chi phí?
2. Có thể dùng cơ chế loại khối lượng xác suất thất bại của CARS để xấp xỉ V* theo reference của MDM-VGB và thay process verifier, trong khi giữ nguyên backbone và kernel VGB, hay không?

Hai đầu ra bắt buộc là **so sánh CARS-QM9 với MDM-VGB-QM9** và **MDM-VGB dùng CARS làm bộ xấp xỉ giá trị**. Nhánh q_rho và toy joint phục vụ kiểm chứng trung gian, không thay thế đầu ra thứ hai. Nếu mới ước lượng được V_rho hoặc V* của pi_rho, phần xấp xỉ V* theo native reference vẫn chưa hoàn thành.

| Phần cần kiểm tra | Đối chứng và điều kiện |
| --- | --- |
| MDM-VGB trên QM9 | Đối chiếu code/settings gốc; giữ riêng kết quả tương thích upstream với biến thể chuẩn hóa của dự án. |
| CARS trên QM9 | CARS được chuyển sang cùng backbone QM9; đối chiếu RS/ARS trên cùng luật sinh. Không gọi đây là tái lập thí nghiệm acrylate/isocyanate của CARS. |
| CARS bên trong MDM-VGB | Chỉ thay nguồn giá trị qua verifier interface; kernel, checkpoint, reward, threshold và cách tính chi phí giữ chung với đối chứng. |
| Giá trị cần xấp xỉ | Khóa reference native và định nghĩa conditional value, kiểm tra root/prefix/mask có lỗ; báo riêng V_rho và V* của pi_rho. |
| Kết quả chính | Chất lượng phân tử theo chi phí thực và độ chính xác giá trị theo cùng target; không chỉ kiểm tra code chạy được. |

CARS là thuật toán lấy mẫu, không phải một dataset QM9 riêng. Benchmark phân tử gốc của CARS dùng các lớp acrylate, chain extender và isocyanate. Dự án này sẽ chuyển cơ chế CARS sang QM9; kết quả của phiên bản chuyển đổi phải được ghi rõ là CARS thích nghi cho QM9.

Tài liệu này mô tả kế hoạch, không phải bằng chứng thực nghiệm. Tiến độ thực tế được ghi trong [PAPER_CHECK.md](PAPER_CHECK.md): đã chuẩn bị assets và thử train backbone, chưa có benchmark so sánh hoặc bằng chứng phương pháp mới vượt baseline. Việc cập nhật kế hoạch không khởi chạy training hoặc benchmark.

## 2. Tài liệu và code làm cơ sở

- [CARS, §2–4 và Appendix E/F/H](https://arxiv.org/html/2510.01902v2): phân phối bị ràng buộc, trie, khối lượng còn lại và thí nghiệm phân tử. Bản PDF được cung cấp nằm trong thư mục resource.
- [MDM-VGB, §3–5 và Appendix B/F](https://arxiv.org/html/2606.28301v1): định nghĩa giá trị, verifier, QM9 và giao thức đánh giá. Appendix B nêu điều kiện reference consistency để regression học đúng giá trị lý tưởng.
- [CARS logits processor](resource/cars-main/cars/logits_process.py): xác suất gốc, cập nhật trie và reweighting.
- [QM9 harness](resource/MDM-VGB-main/src/tasks/qm9/harness.py): mô hình, logits và rollout MDLM.
- [QM9 task](resource/MDM-VGB-main/src/tasks/qm9/task.py): dữ liệu, QED, reward và metrics.
- [QM9 verifier](resource/MDM-VGB-main/src/tasks/qm9/verifier.py): giao diện đánh giá một hoặc nhiều trạng thái.
- [VGB candidate construction](resource/MDM-VGB-main/src/algorithms/vgb_candidates.py) và [terminal handling](resource/MDM-VGB-main/src/algorithms/vgb_terminal.py): điểm tích hợp và chính sách ngân sách.

Giữ các repository trong resource làm nguồn tham chiếu, không sửa và không import trong chương trình. Khi cần dùng lại code, sao chép phần cần thiết sang package của dự án, giữ thông tin nguồn và license, rồi sửa các import cho độc lập; phần còn lại được viết mới. Không thêm resource vào sys.path, không cài repository trong resource làm dependency và không đóng gói resource vào image Modal. Không xem các file đang mở trong IDE là đã tồn tại: tại thời điểm lập kế hoạch ban đầu, workspace chỉ có thư mục resource cùng metadata.

## 3. Mô hình nền và dữ liệu dùng chung

### 3.1. Mô hình

Dùng QM9 MDLM của MDM-VGB làm backbone chung, đóng băng trọng số trong toàn bộ thí nghiệm inference. Paper mô tả backbone khoảng 92,4 triệu tham số, dùng kiến trúc UDLM-QM9 với objective MDLM.

| Thành phần | Lựa chọn đã viết trong source/config |
| --- | --- |
| Kiến trúc | `src/models/configuration_udlm.py` và `src/models/modeling_udlm.py`, nguồn được ghi trong THIRD_PARTY_NOTICES.md |
| Dataset | yairschiff/qm9 |
| Tokenizer | yairschiff/qm9-tokenizer |
| Chiều dài | 32 token |
| Checkpoint inference | Manifest cụ thể dưới `workspace/model_data/qm9/<run-id>/attempts/<attempt-id>/checkpoints/step-<step>/`; không tự chọn latest |
| Base training | Khởi tạo từ đầu theo cấu hình QM9 upstream, seed 1, 25.000 bước |
| Inference | Cùng checkpoint, tokenizer, precision và quy tắc token hợp lệ giữa các phương pháp |

`model_data/qm9/mdlm_qm9/last` là đường dẫn tham chiếu của upstream, không phải pretrained model được tải bởi dự án. Prepare assets chỉ tải dataset/tokenizer; `train_base.py` huấn luyện model từ đầu hoặc resume checkpoint dự án có provenance. PAPER_CHECK.md ghi nhận attempt Modal đã lưu checkpoint 5.000/10.000 bước trước khi mất kết nối; hoàn tất 25.000 bước và trạng thái job sau đó chưa được xác minh. Không suy ra đã có model cuối chỉ từ mô tả kiến trúc của paper.

Assets đã chuẩn bị ghi nhận vocabulary 35 token với ID 0..34 và split 127.190/6.695. Fast tokenizer giữ regex/ID trong vocabulary đã công bố, sửa lỗi cộng special-token offset thêm lần nữa của class cũ; checkpoint dùng offset cũ không tương thích. Runtime đối chiếu token rules và hash, không suy ra compatibility chỉ từ số 35 trong paper hoặc số 40 trong config verifier. Dependency chưa có lock file đã kiểm chứng.

### 3.2. Phân chia dữ liệu

- Giữ split upstream với tỷ lệ held-out 5%, seed split 42; lưu danh sách ID và hash để các lần chạy dùng đúng cùng split.
- Chỉ huấn luyện base model trên phần train. Chia train thành các nhóm phục vụ training và validation verifier, tránh tách các snapshot của cùng một rollout sang hai nhóm khác nhau.
- Tạo một tập calibration bằng các lần sinh độc lập từ base model đã đóng băng. Tập này không được dùng làm tập đánh giá cuối hoặc để chọn siêu tham số verifier.
- Tạo panel trạng thái riêng cho kiểm chứng giá trị, gồm snapshot base, trạng thái có mask không liên tiếp và trạng thái được VGB ghé thăm.
- Dùng phần held-out để kiểm tra dữ liệu và đánh giá tổng quát hóa khi phù hợp. QM9 generation là bài toán vô điều kiện: không dùng SMILES held-out làm prompt rồi gọi đó là sinh vô điều kiện.

Canonicalize cả train và output bằng cùng phiên bản RDKit khi đo novelty hoặc deduplication. Giữ cả token gốc và SMILES sau chuẩn hóa để kiểm tra lại.

### 3.3. Reward và token đặc biệt

Đặt điểm phân tử R(y) bằng QED nếu phân tử hợp lệ, bằng 0 nếu không hợp lệ. Reward chính là:

\[
\tau(y)=\mathbf{1}\{y\text{ hoàn chỉnh, hợp lệ và }\mathrm{QED}(y)\ge T\}.
\]

T là quantile 95% của điểm trên tập base calibration độc lập; giữ nguyên T cho mọi phương pháp trong cùng một giao thức. Báo cáo số lượng calibration, sampler, seed, cách tính quantile và bất định của threshold. Threshold 0.5614854131280362 của upstream là thông tin tham chiếu; source dự án lấy threshold từ completed calibration manifest. Không thay calibration bằng percentile QED của dataset. Calibration chưa chạy nên chưa có threshold dự án được xác minh.

Chốt chung quy tắc BOS, EOS, PAD, UNK, MASK và giới hạn 32 token trước khi lấy mẫu. Trạng thái còn mask không được trở thành thành công chỉ vì hàm decode xóa token mask. Grammar phải tương thích với cùng cách diễn giải EOS/PAD và không loại bỏ các SMILES hợp lệ thuộc target. Báo cáo riêng phân tử hợp lệ nằm ngoài miền QM9, chẳng hạn vượt chín heavy atom; chỉ thêm ràng buộc miền vào reward trong một thí nghiệm được khai báo riêng.

## 4. Phân biệt giá trị cần ước lượng

### 4.1. Giá trị lý tưởng trong paper

Với C(z) là tập các chuỗi hoàn chỉnh khớp mọi vị trí đã biết của z:

\[
V^*(z)=\mathbb{E}_{\pi_{\mathrm{ref}}}[\tau(Y)\mid Y\in C(z)].
\]

Muốn khẳng định đang xấp xỉ V*, cần xác định rõ phân phối joint pi_ref và luật điều kiện trên C(z). Chỉ dùng chung trọng số MDLM chưa đủ để chứng minh mọi thứ tự reveal và native diffusion rollout biểu diễn cùng một phân phối.

### 4.2. Giá trị rollout khả thi ban đầu

Chọn thứ tự rho cố định, ban đầu từ trái sang phải trên các vị trí còn mask. Giữ nguyên token đã biết của z và dùng logits MDLM để lần lượt điền các vị trí còn lại. Gọi phân phối hoàn tất thu được là q_rho(y | z), và định nghĩa:

\[
V_\rho(z)=\mathbb{E}_{q_\rho(\cdot\mid z)}[\tau(Y)].
\]

Đây là target trực tiếp của estimator CARS ban đầu. Đặt tên rõ V_rollout hoặc V_rho trong cấu hình, artifact và báo cáo. Chỉ đồng nhất với V* khi có cơ sở về reference consistency. Không coi việc hai thứ tự cho kết quả gần nhau là bằng chứng đầy đủ về consistency.

Giữ riêng hai kết luận: độ chính xác khi ước lượng V_rho và tác dụng thực nghiệm khi dùng giá trị đó trong native MDM-VGB.

### 4.3. Target bắt buộc cho nhánh tích hợp chính

Khóa một giao thức native BASE từ root, gồm checkpoint, số diffusion step, schedule và cách xử lý endpoint, rồi gọi luật terminal của giao thức đó là pi_native. Khi đó target vận hành được xác định rõ là:

\[
V^*_{\mathrm{native}}(z)=
\frac{\Pr_{\pi_{\mathrm{native}}}(Y\in C(z),\tau(Y)=1)}
     {\Pr_{\pi_{\mathrm{native}}}(Y\in C(z))}.
\]

Giá trị chỉ được định nghĩa khi mẫu số dương. Đây là việc cụ thể hóa reference cho thí nghiệm, không phải chứng minh logits MDLM tại mọi mask là conditional chính xác của joint này. Chạy native completion mới từ z cũng chưa mặc nhiên lấy mẫu từ pi_native điều kiện trên C(z). Giữ phân biệt native inference 32-step, nguồn label trajectory 128-step của upstream và reference pi_rho; không đổi tên một reference thành reference khác để vượt cổng kiểm chứng.

Neural verifier upstream vẫn là đối chứng thực nghiệm của paper. Khi cần so sánh estimator cùng target V*_{native}, bổ sung đối chứng neural có label cùng reference: lấy Y từ native root rồi chọn mask pattern độc lập với Y và gán tau(Y). Tính độc lập này cho conditional regression target đúng theo pi_native; đây là control bổ sung của dự án, không phải cách lấy snapshot của paper. Ghi riêng kết quả của hai neural verifier.

## 5. Thiết kế estimator CARS

### 5.1. Trie và pruning an toàn

Mỗi cây hoàn tất gắn với trạng thái gốc z và policy rho. Node lưu xác suất token gốc, các child đã biết và khối lượng xác suất còn lại; dùng tính toán log-space ổn định khi cần.

- Khởi đầu chưa biết thất bại: khối lượng còn lại là 1.
- Nhánh được chứng minh không chứa bất kỳ completion thành công nào: khối lượng là 0.
- Node nội bộ: khối lượng là tổng xác suất cạnh gốc nhân khối lượng child.
- Không cộng lại khối lượng các failure cone chồng lấp; loại trùng và bảo toàn quan hệ cha–con.
- Cache xác suất gốc theo đúng model, trạng thái và policy. Không dùng xác suất sau reweighting làm xác suất backbone.

Đặt W là tập nhánh thất bại được chứng nhận, A_W là sự kiện tránh W và:

\[
B_W(z)=\Pr_{q_\rho}(A_W\mid z).
\]

Nếu mọi nhánh bị loại đều có reward 0, thì V_rho(z) không vượt B_W(z). Cận này có thể rất lỏng; nó không có bảo đảm sai số nhân như giả định verifier trong lý thuyết VGB.

Triển khai từ trường hợp đơn giản tới phức tạp: loại các lá thất bại đã quan sát trước, sau đó bổ sung pruning theo grammar được kiểm chứng. Khi chỉ loại lá, ghi rõ đây là mức ARS của cơ chế thích nghi; không gán hiệu quả của nó cho full CARS.

Oracle grammar chỉ được loại một nhánh khi chắc chắn không thể hoàn tất thành công trong giới hạn chiều dài và với token đã cố định. Với mask có lỗ, kiểm tra một prefix liên tiếp thực sự hoặc một oracle hỗ trợ ràng buộc vị trí; không ghép các token đã reveal thành một chuỗi rồi kiểm tra như prefix. RDKit không parse được một SMILES chưa hoàn chỉnh không phải là chứng nhận thất bại. QED thấp ở một lá chỉ cho phép loại lá đó nếu chưa có một cận QED an toàn cho cả nhánh.

### 5.2. Học trie rồi đóng băng để ước lượng

Tách mỗi lượt đánh giá thành hai phần: discovery cập nhật W, và estimation dùng W đã đóng băng với các draw mới, độc lập. Mọi chi phí của discovery được ghi nhận.

Phân phối proposal sau pruning là:

\[
Q_W(y\mid z)=\frac{q_\rho(y\mid z)\mathbf{1}_{A_W}(y)}{B_W(z)}.
\]

Với m draw đầy đủ từ Q_W, trước bước chấp nhận theo reward, estimator đề xuất là:

\[
\widehat V_\rho(z)=B_W(z)\frac{1}{m}\sum_{j=1}^{m}\tau(Y_j).
\]

Đây là suy luận thiết kế của dự án từ cơ chế CARS, không phải estimator được tuyên bố sẵn trong paper. Trong điều kiện pruning an toàn và sampling đúng Q_W, estimator không chệch có điều kiện theo trie đã đóng băng. Nếu B_W bằng 0 do loại hết support, trả giá trị 0 với chứng nhận; nếu do lỗi số học, đánh dấu lỗi thay vì coi là chứng nhận.

Ghi nhận cả draw thành công và thất bại; không ước lượng từ các mẫu cuối đã được CARS chấp nhận. Dùng số draw cố định cho từng cấu hình và confidence interval nhị thức được nhân với B_W. Trường hợp không thấy thành công vẫn cần upper confidence bound; không suy ra giá trị thật bằng 0. Nếu hết ngân sách trước số draw đã định, ghi nhận estimator chưa đủ dữ liệu và áp dụng fallback đã khai báo, không tự nhận một ước lượng không chệch sau stopping phụ thuộc kết quả.

Chọn discovery budget và m trên pilot riêng. So sánh với Monte Carlo không pruning ở cùng tổng số forward evaluation, không chỉ cùng số draw estimation.

### 5.3. Kiểm chứng reference và đóng khoảng trống tới V* native

**Joint pi_rho là kiểm chứng trung gian.**

Bổ sung một chế độ kiểm chứng dùng joint pi_rho(y) được định nghĩa bằng việc sinh toàn chuỗi từ root theo cùng một thứ tự cố định. Mỗi xác suất token trong joint được tính từ prefix đã reveal và phần còn lại còn mask, nên joint có likelihood tính được. Đây là reference kiểm chứng được chỉ định rõ, không mặc định là phân phối của native diffusion sampler.

Trên trạng thái có mask bất kỳ, q_rho(y | z) thường khác pi_rho(y | C(z)). Khi cần target chính xác đối với joint này, dùng importance weight:

\[
w_z(y)=\frac{\pi_\rho(y)}{q_\rho(y\mid z)}.
\]

Khi proposal phủ toàn bộ support tương thích, ta có:

\[
V^*_{\pi_\rho}(z)=\frac{\mathbb{E}_{q_\rho}[w_z(Y)\tau(Y)]}{\mathbb{E}_{q_\rho}[w_z(Y)]}.
\]

CARS có thể giảm chi phí ước lượng tử số bằng B_W nhân trung bình w_z tau dưới Q_W. Mẫu số phải được ước lượng từ proposal q_rho chưa pruning, hoặc tính chính xác: những completion thất bại bị loại vẫn đóng góp vào xác suất C(z). Không dùng riêng các draw CARS còn lại cho cả tử và mẫu.

Kiểm tra công thức bằng enumeration trên joint nhỏ trước. Ratio estimator nhất quán theo số mẫu nhưng có thể chệch ở cỡ mẫu hữu hạn. Báo cáo effective sample size, weight concentration, khoảng bất định và chi phí scoring joint; không âm thầm clip weight. Với prefix hợp lệ theo rho, kiểm tra công thức rút về estimator đơn giản ở mục 5.2.

Chế độ này kiểm chứng V* của pi_rho, chưa đáp ứng V*_{native}. Dùng giá trị đã hiệu chỉnh cùng logits native cũng không tạo ra bảo đảm lấy mẫu chính xác từ joint pi_rho, vì transition conditionals còn cần khớp reference.

**Nhánh native là yêu cầu chính: source đã viết ở Step 9, kiểm chứng thực thi và khả thi QM9 còn chờ.**

Một hướng kiểm chứng trực tiếp là biểu diễn toàn bộ trajectory native H, từ root tới terminal Y(H), với luật P_native(H). Không cần coi likelihood của một completion trái sang phải là likelihood native. Với E_z(H)=1[Y(H) thuộc C(z)], tính:

\[
N_z=\mathbb E_{P_{\mathrm{native}}}[E_z(H)\tau(Y(H))],\qquad
D_z=\mathbb E_{P_{\mathrm{native}}}[E_z(H)],\qquad
V^*_{\mathrm{native}}(z)=N_z/D_z.
\]

CARS có thể reweight cây trajectory để ước lượng N_z: chỉ loại nhánh được chứng nhận có E_z tau=0, đóng băng W, rồi lấy B_W nhân trung bình E_z tau từ proposal còn lại. D_z cần draw native chưa pruning theo reward hoặc một estimator có hiệu chỉnh được chứng minh; không dùng riêng các trajectory CARS được chấp nhận để tính mẫu số. Đây là thiết kế bổ sung của dự án từ định nghĩa conditional expectation, không phải thuật toán được hai paper triển khai sẵn.

Nếu dùng proposal trajectory có ràng buộc quan sát để giảm sự kiện hiếm, phải giữ likelihood ratio P_native/Q_z, kiểm chứng support và tính cả scoring cost. Việc điền sẵn token z từ đầu rồi gọi native sampler chưa đủ: nó làm thay đổi lịch sử context và không tự cung cấp conditional law cần thiết. Khi tuyến tính hóa các token draw trong cùng diffusion step, giữ logits từ trạng thái đầu step để không âm thầm đổi native transition.

Kiểm chứng N_z, D_z, mass và ratio bằng enumeration trajectory nhỏ trước khi chọn phương án QM9. Ratio hữu hạn mẫu có thể chệch; nếu mẫu số không được phân giải, ESS thấp hoặc chi phí quá lớn, báo unavailable/giới hạn và chưa xác nhận phần V* native hoàn thành. V_rho có thể được báo như surrogate ablation, nhưng không dùng nó để đổi mục tiêu nghiên cứu. Không suy ra stationary-law guarantee chỉ từ việc estimator đúng target.

### 5.4. Cache và tính ổn định

Khóa cache gồm toàn bộ token của z, checkpoint, tokenizer, reward/threshold, policy rho, quy tắc token đặc biệt và cấu hình estimator. Không coi hai trạng thái có cùng prefix nhưng token tương lai khác nhau là một trạng thái.

Đóng băng giá trị đã hoàn tất của một trạng thái trong mỗi run, để cùng một trạng thái không thay đổi giá trị do một truy vấn mới. Source dùng LRU cho trie resident và giữ compact value đã hoàn tất trong run; eviction trie không xóa giá trị đó. Snapshot disk là tùy chọn, mặc định tắt, và warm load phải khớp full context. Mỗi cold attempt có scope run/attempt riêng; benchmark không dùng lại cache giữa seed hoặc lần thử lại. Online update giá trị đã hoàn tất vẫn là ablation tương lai, không kế thừa stationary-law guarantee của kernel cố định.

## 6. Tích hợp và bố trí source

| Khu vực | Trách nhiệm |
| --- | --- |
| src/models | Kiến trúc độc lập, adapter model/tokenizer và thống kê forward calls |
| src/qm9 | Split, canonicalization, terminal reward, calibration và rollout |
| src/sampling | Native sampling, completion/reference, Base, BoN, rejection, VGR, VGB và Momentum |
| src/cars | Trie, oracle, proposal và estimator giá trị |
| src/verifiers | Adapter CARS, Monte Carlo, neural verifier và importance mode qua cùng giao diện |
| src/training | Training base model và neural verifier |
| src/experiments | Run, seed, budget, panel giá trị và điều phối benchmark |
| src/evaluation | Đọc artifact, tính metrics, tổng hợp bất định và vẽ đồ thị |
| modal | Mỗi task một file Python và một script shell gọi modal run |
| configs | Reference mode, reward, estimator, sampler và budget |
| tests | Các kiểm chứng có ý nghĩa về xác suất, pruning, estimator và tích hợp |
| workspace/results/raw, value_checks, benchmarks và summary | Artifact generation/verification, kiểm tra giá trị, điều phối và bảng/đồ thị |

11 cặp Python/shell Modal dưới đây đã được viết; `run.py` chọn runner local tương ứng. Cây source/artifact đầy đủ và checklist bằng chứng nằm trong implementation.md. Việc có source không xác nhận task đã chạy hoặc gate đã qua.

```text
sampling/
├── PLAN.md
├── modal/
│   ├── README.md                       # Danh sách task, đầu vào/đầu ra và cách chạy
│   ├── prepare_assets.py               # Chuẩn bị QM9, tokenizer, split và manifest
│   ├── run_prepare_assets.sh           # Chạy prepare_assets bằng modal run
│   ├── train_base.py                   # Train/resume backbone MDLM QM9
│   ├── run_train_base.sh               # Chạy train_base bằng modal run
│   ├── calibrate.py                    # Sinh calibration và khóa threshold QED
│   ├── run_calibrate.sh                # Chạy calibrate bằng modal run
│   ├── collect_rollouts.py             # Tạo snapshot và label cho neural verifier
│   ├── run_collect_rollouts.sh         # Chạy collect_rollouts bằng modal run
│   ├── train_verifier.py               # Train/resume neural verifier theo target
│   ├── run_train_verifier.sh           # Chạy train_verifier bằng modal run
│   ├── prepare_value_panel.py          # Tạo panel trạng thái dùng chung để kiểm chứng
│   ├── run_prepare_value_panel.sh      # Chạy prepare_value_panel bằng modal run
│   ├── generate.py                     # Sinh mẫu theo method, verifier, config và seed
│   ├── run_generate.sh                 # Chạy generate bằng modal run
│   ├── check_values.py                 # Kiểm chứng estimator trên toy joint/panel QM9
│   ├── run_check_values.sh             # Chạy check_values bằng modal run
│   ├── benchmark.py                    # Điều phối ma trận phương pháp, budget và seed
│   ├── run_benchmark.sh                # Chạy benchmark bằng modal run
│   ├── evaluate.py                     # Tổng hợp metrics, bảng, đồ thị và tải báo cáo
│   ├── run_evaluate.sh                 # Chạy evaluate bằng modal run
│   ├── verify.py                       # Kiểm chứng xác suất, reward, cache và accounting
│   └── run_verify.sh                   # Chạy verify bằng modal run
├── configs/
│   └── qm9.yaml                        # Một YAML thường cho dữ liệu/reward QM9
└── src/
    ├── __init__.py                     # Khai báo package src
    └── config.py                       # Đọc YAML thường và kiểm tra đường dẫn
```

Modal là môi trường chạy các task của dự án ngay từ đầu. Các kiểm chứng nhỏ vẫn có thể chạy local, và cũng có task verify để chạy chúng trong image Modal. Logic khoa học nằm trực tiếp trong package src, khai báo bằng src/__init__.py; các file modal chỉ cấu hình tài nguyên, nhận tham số và gọi logic tương ứng. Dùng import theo namespace src.qm9, src.cars, src.sampling và các subpackage tương ứng. pyproject.toml đóng gói src cùng subpackage; image Modal giữ project root để import nhất quán. Entry point local dự kiến là python run.py --task <task> từ repo root, mỗi task có entrypoint Modal tương ứng.

Theo template Nam cung cấp, mỗi file modal/<task>.py khai báo trực tiếp App, Image.from_registry, pip dependencies/files, Volume.from_name và @app.function run_command; giữ main cho modal run và khối __main__ với app.run() cho python. Image dùng tag nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 và Python 3.11; CPU/GPU, memory, timeout và Volume điền trực tiếp trong từng file task. Dự án chỉ dùng QM9; cấu hình khoa học đọc từ YAML thường thành dictionary, nguồn setting ghi bằng comment và [PAPER_CHECK.md](PAPER_CHECK.md).

Mỗi task có cặp modal/<task>.py và modal/run_<task>.sh. Script gọi modal run modal/<task>.py::main, chuyển tiếp tham số, giữ exit code và xác định project root để không phụ thuộc thư mục gọi lệnh. Đây là quy ước cho bước triển khai sau, chưa tạo file thực thi trong STEP 4. CLI chính thức hỗ trợ tham chiếu file::entrypoint: [Modal run](https://modal.com/docs/cli/latest/run).

Danh sách task được chốt trong [modal/README.md](modal/README.md): prepare_assets, train_base, calibrate, collect_rollouts, train_verifier, prepare_value_panel, generate, check_values, benchmark, evaluate và verify. File generate nhận method/config để chạy từng phương pháp; benchmark điều phối các run theo ma trận đã khóa. Cấu hình tài nguyên ghi trực tiếp trong file Modal của từng task, tách khỏi siêu tham số thuật toán. Không tạo modal/__init__.py để tránh che package SDK modal khi chạy từ project root.

Dataset, tokenizer, checkpoint, snapshot trie và kết quả cần giữ giữa các task được lưu trên một Modal Volume, mount tại /root/workspace với các thư mục data, model_data, cache/tries và results. Image đặt project root tại /root; đường dẫn artifact tương đối vẫn có tiền tố workspace/ như local. Các task dùng ID và manifest để tham chiếu đúng artifact. Checkpoint model được tái sử dụng; thay verifier không kích hoạt training backbone lại. Snapshot trie chỉ được nạp nếu metadata khớp đầy đủ khóa cache ở mục 5.4. Triển khai commit sau khi ghi và reload trước khi đọc artifact mới từ task khác; mỗi run/attempt có đường dẫn riêng, tránh nhiều worker ghi cùng file. Cơ chế lưu trữ theo [Modal Volumes](https://modal.com/docs/guide/volumes).

Image chỉ đưa vào các file cần chạy từ package độc lập, configs, entrypoint Modal, metadata dependency/license và tests cùng fixtures cho task verify. Không mount hoặc upload resource. Task evaluate lưu báo cáo vào Volume và hỗ trợ tải các artifact được chọn về results ở máy local; không tải lại toàn bộ dataset/checkpoint sau mỗi run.

Adapter verifier CARS sẽ hỗ trợ chấm theo batch và trả giá trị theo đúng thứ tự trạng thái đầu vào. Dùng lại điểm tích hợp của QM9Verifier. Trạng thái hoàn chỉnh được neo bằng terminal reward chính xác; các thống kê độ bất định, cache và chi phí được lưu riêng.

Không sửa backbone khi thay verifier. Trong so sánh neural–MC–CARS, giữ nguyên graph, cách shortlist, seed policy và chính sách terminal; chỉ thay nguồn giá trị. Điểm tích hợp là adapter `src/verifiers/cars.py` trả ordered `ValueEstimate` qua `values_for_states` cho candidate builder; CARS không thay generator bằng một luật khác bên trong outer VGB. Nhánh rho có neural-rho để kiểm chứng cùng target; nhánh chính cần estimator native và control neural-native-conditional ở mục 4.3. Giữ verifier snapshot upstream riêng để phân biệt tái lập paper với control mới.

## 7. Baseline và ma trận so sánh

| Nhóm | Phương pháp | Câu hỏi |
| --- | --- | --- |
| Native QM9 | Base, BoN | Chất lượng unguided và baseline outcome-only |
| Native QM9, tham chiếu upstream | VGR, VGB, VGB-Momentum với learned verifier upstream | Mốc theo giao thức MDM-VGB; sai khác implementation phải được đối chiếu trước |
| Cùng q_rho | Base-rho, rejection sampling, ARS, CARS | Giá trị của adaptive pruning trên cùng luật sinh |
| Cùng graph VGB, kiểm chứng trung gian | Neural-rho, MC-rho, CARS-rho verifier | Độ đúng và hiệu quả theo V_rho, chưa phải hoàn thành nhánh V* native |
| Joint pi_rho, kiểm chứng trung gian | Exact values, MC và CARS có importance correction | Tách rõ V_rho và conditional V* của joint trái sang phải |
| Cùng graph VGB, nhánh chính bắt buộc | Neural-native-conditional, MC-native-conditional, CARS-native-conditional; thêm upstream-neural làm mốc riêng | CARS có xấp xỉ V*_{native} và cải thiện chất lượng/chi phí khi chỉ thay verifier không? |

Không gộp native Base và Base-rho thành một baseline, dù dùng chung trọng số. Không dùng Llama/Qwen từ CARS gốc làm đối chứng chính với QM9 MDLM vì như vậy thay đổi cả model và sampler.

Thiết lập ban đầu bám cấu hình upstream: chiều dài 32, L_f = 8, L_b = 32, K = 8, singleton updates, lambda = 0 và chi = 1. Kiểm tra sensitivity với lambda và reveal order sau khi đường chạy cơ bản đúng. Giữ các default được kiểm tra từ code tách biệt với tuyên bố tái tạo đúng mọi kết quả paper.

Chạy pilot riêng để đo toàn bộ chi phí lồng và uncertainty của từng nhánh, đặc biệt mẫu số của native conditional value. Các con số 128 output pilot hoặc 3 seed × 1.000 output đã đề xuất trước đây chưa được chốt và không phải setting paper đã xác minh. Khóa số seed/output, grid N, discovery budget và draw count sau profiling, trước khi xem tập đánh giá cuối; dùng setting có nguồn từ paper/code khi có, ghi rõ lựa chọn dự án khi tài liệu không chỉ định. Không tự giảm số draw theo kết quả reward để làm query vừa cap.

Editing từ leaf có reward thấp là phần mở rộng sau generation, dùng cùng pool seed cho mọi phương pháp. Không dùng kết quả generation khác nhau của mỗi phương pháp làm đầu vào repair rồi so sánh trực tiếp.

## 8. Ngân sách và metrics

### 8.1. Tính đủ chi phí

Ngân sách n nhân N chỉ là mốc bước của VGB, không phải toàn bộ chi phí khi verifier tự gọi backbone. Cần đếm tại ranh giới forward thật, gồm:

- Forward của sampler ngoài và mọi rollout lồng trong MC/CARS.
- Discovery, estimation, scoring likelihood và fallback completion.
- Số lượt neural verifier, số trạng thái được chấm và batch size.
- Chi phí oracle RDKit/grammar, cập nhật trie, cache miss, CPU/GPU memory và wall-clock time.

Báo cáo cả số lần gọi batch thực và số lượt đánh giá từng sequence, để batching không làm sai so sánh. Dùng FLOP-adjusted NFE khi đo được tỷ lệ chi phí verifier/base; không tự lấy tỷ lệ từ số tham số. Ghi riêng chi phí offline training/calibration và inference, cùng kết quả cache cold/warm. Tính cả chi phí warm-up trong chi phí amortized trên số output đã công bố.

Tách hai giao thức: tham chiếu paper với step budget và completion được tính thêm; so sánh chính với cap tổng forward-work đã định trước. Trong cap tổng, dự phòng chi phí một completion unguided. Chỉ bắt đầu nhóm query verifier khi còn đủ ngân sách cho discovery và toàn bộ số draw cố định theo cận chi phí đã tính; nếu không đủ, chuyển ngay sang completion unguided thay vì trả một giá trị giả. Nếu không thể hoàn tất trong cap, ghi timeout/unfinished và tính là thất bại. Không dùng adaptive stopping chưa phân tích để cứu một estimator đang thiếu mẫu.

### 8.2. Chất lượng output

- Pass@95 trên toàn bộ output được yêu cầu, gồm invalid, timeout và unfinished trong mẫu số.
- Validity của output hoàn chỉnh; tỷ lệ chưa hoàn tất và timeout riêng.
- Mean/median QED trên valid output, kèm QED trung bình trên mọi output với invalid bằng 0.
- Unique Pass@95 bằng số phân tử canonical khác nhau vượt threshold chia tổng số output yêu cầu.
- Uniqueness và novelty, ghi rõ mẫu số là tổng output hay valid output.
- Diversity qua Morgan fingerprints và Tanimoto distance, dùng cùng cấu hình và chính sách subsampling cặp.
- Quality–cost frontier và chi phí để đạt một số lượng phân tử thành công, unique đã định trước.

Không loại duplicate khỏi các mẫu dùng để ước lượng V: deduplication sẽ đổi phân phối sampling. Deduplicate chỉ trong metrics yêu cầu tính unique.

### 8.3. Độ chính xác giá trị

Đánh giá theo số mask còn lại, dạng mask và nguồn trạng thái: sai số tuyệt đối, RMSE/Brier, reliability, confidence-interval coverage và chi phí trên mỗi query. Với reference có importance weights, thêm ESS và concentration của weight.

Ground truth chính xác chỉ đến từ enumeration trên không gian nhỏ hoặc trạng thái ít mask có reference joint tính được. Monte Carlo nhiều mẫu trên QM9 là giá trị tham chiếu có bất định, không được gọi là ground truth chính xác. Panel chính cần đánh giá conditional value theo pi_native; đo riêng uncertainty của mẫu số/importance weights. Panel rho là kiểm chứng bổ sung. Nếu so surrogate V_rho với V*_{native}, phải ghi rõ đây là tổng approximation error gồm target mismatch, không gọi đó là riêng sai số sampling của CARS.

## 9. Các bước triển khai theo thứ tự

1. **Chốt giao thức và môi trường:** xác minh dependency, assets, model/tokenizer và các quy tắc trạng thái; lưu cấu hình và manifest dữ liệu.
2. **Chuẩn bị backbone:** lấy checkpoint có provenance hoặc huấn luyện đúng cấu hình; đóng băng và kiểm tra base generation.
3. **Xây reward và calibration:** hoàn thiện terminal checks, tạo calibration độc lập, khóa threshold và panel đánh giá.
4. **Tái tạo baseline:** native Base/BoN/VGR/VGB/Momentum, rollout và learned verifier; ghi rõ sai khác với paper.
5. **Xác định q_rho và MC-rho:** tạo đường hoàn tất cố định, đo sự phụ thuộc reveal order và dựng baseline giá trị không pruning.
6. **Xây trie và estimator:** bắt đầu với leaf rejection, xác minh khối lượng, thêm grammar pruning an toàn, rồi frozen-trie estimation.
7. **Đóng khoảng trống V* native:** pi_rho importance chỉ là kiểm chứng trung gian; khóa native reference, kiểm chứng estimator trajectory/conditional và control cùng target bằng enumeration, sau đó profiling QM9. Không coi bước này là tùy chọn nếu muốn hoàn thành mục tiêu tích hợp.
8. **Tích hợp verifier:** dùng cùng VGB graph, accounting và policy cho neural/MC/CARS native-conditional; giữ terminal anchoring. Nhánh rho và upstream-neural vẫn có nhãn riêng.
9. **Pilot và khóa grid:** đo chi phí lồng, chọn ngân sách khả thi bằng tập pilot, ghi lựa chọn trước benchmark cuối.
10. **Benchmark và phân tích:** chạy seed đã chốt, tạo bảng và frontier, báo cáo lợi ích hoặc không có lợi ích cùng giới hạn thống kê.

Mỗi bước phụ thuộc vào kiểm chứng của bước trước; không dùng kết quả QM9 để bỏ qua một lỗi về support, reward hoặc khối lượng trie.

## 10. Kiểm chứng cần hoàn thành

### 10.1. Xác suất và trie

Trên joint nhỏ có thể liệt kê đầy đủ, kiểm tra tổng xác suất bằng 1, khối lượng trie bằng tổng trực tiếp, cập nhật không tăng B_W, failure trùng không bị trừ hai lần và không có successful leaf bị prune. Kiểm tra tách biệt khối lượng còn lại với giá trị thật và xác minh phân phối proposal reweighted.

Trên grammar có support liệt kê được, dùng nhiều lượt chạy độc lập để so sánh phân phối accepted sample của CARS với phân phối conditioned chính xác. Đây là kiểm chứng triển khai, không phải chứng minh exactness từ một empirical KL thấp. Không dùng KL tới base chưa conditioned làm thước đo fidelity tới target conditioned.

### 10.2. Estimator và reference consistency

Kiểm tra frozen-trie estimator ở root, prefix và arbitrary masks; các trường hợp reward toàn 0/toàn 1, B_W bằng 0, không quan sát thành công và trie không prune gì. Với arbitrary masks của một joint nhất quán, kiểm tra Bellman identity theo mọi vị trí và so sánh các reveal order. Với neural MDLM, báo cáo Bellman residual và độ nhạy theo order như diagnostics.

Đối với importance mode, kiểm tra tử số/mẫu số bằng enumeration, likelihood của joint, proposal support và sai số ratio qua nhiều seed. Không tuyên bố exactness của native masked-state chain từ các kiểm chứng trên reference cố định.

Đối với nhánh native, thêm enumeration trajectory gồm reveal timing/MASK self-loop và mọi quan sát tương lai. Đối chiếu joint terminal suy ra từ trajectory, N_z, D_z, CARS proposal và conditional value trực tiếp. Kiểm tra rằng logits trong một bước song song không bị cập nhật theo token vừa chọn, và label control dùng mask độc lập với terminal. Chưa qua các kiểm chứng này thì không gắn nhãn native V* cho adapter.

### 10.3. Tích hợp, reward và chi phí

Kiểm tra token đã reveal không thay đổi trong completion query, batch output đúng thứ tự, terminal value bằng reward, EOS/PAD đúng quy ước và trạng thái chưa hoàn tất không bị tính thành công. Kiểm tra cache không tái sử dụng qua model hoặc threshold khác.

Đếm forward thật và so với accounting trên một run nhỏ, gồm nested rollout và completion. Kiểm tra timeout không vượt cap đã công bố. Snapshot cùng rollout phải nằm cùng một split verifier; kiểm tra uniqueness/novelty bằng canonical identity ở cả hai phía.

### 10.4. Những điểm upstream phải audit

- Cấu hình inference và rollout hiện dùng số diffusion step khác nhau; giữ khác biệt này có chủ đích hoặc tạo biến thể aligned, không đồng nhất chúng ngầm.
- Đường edge-walk hiện tiếp tục bằng shortlisted forward candidates khi hết step budget; giao thức paper mô tả forward-only completion theo base model. Cần xác minh và đặt policy rõ trước so sánh.
- Trường nfe hiện có thể tăng cho leaf backtrack không gọi model; dùng instrumentation để đo forward thực thay vì tin nguyên counter.
- Terminal acceptance của QM9 hiện kiểm tra validity, trong khi reward là Pass@95; audit tất cả đường thoát, zero-weight candidate và fallback để validity không bị coi là high-reward success.
- Hàm clean SMILES xóa mask và train novelty set dùng chuỗi từ dataset; cần kiểm tra completion trước decode và canonicalize nhất quán.

Giữ kết quả upstream nguyên bản và kết quả theo giao thức đã chuẩn hóa thành hai nhãn nếu các sửa khác biệt làm thay đổi hành vi.

## 11. Artifact và tiêu chí hoàn thành

Mỗi run sẽ lưu cấu hình đã resolve, seed, hash model/tokenizer/dataset/threshold, reference mode, token state, output gốc và canonical, reward, trạng thái completion, cost counters và thông tin estimator. Tách discovery draw khỏi estimation draw; lưu đủ dấu vết để tái tính metrics và kiểm tra khối lượng. Run ID duy nhất và artifact bất biến giúp resume mà không ghi đè kết quả cũ.

Kết quả tổng hợp gồm bảng baseline, quality–cost frontier, error–cost của estimator, cache cold/warm, order sensitivity và confidence interval. Dùng bất định theo seed và cluster rollout khi thích hợp; không coi các snapshot tương quan là mẫu độc lập. Các so sánh chính và mẫu số metrics phải được công bố trước khi chọn kết quả tốt nhất.

Dự án được xem là hoàn thành khi:

- Có backbone và dữ liệu với provenance tái tạo được, calibration độc lập và reward thống nhất.
- Có MDM-VGB-QM9 đã đối chiếu upstream và CARS-QM9 độc lập chạy qua evaluator chung, cùng protocol có provenance; không gọi adaptation QM9 là thí nghiệm molecular gốc của CARS.
- Có MDM-VGB dùng CARS verifier theo target V*_{native} đã xác định, cùng control neural/MC của target này; rho/toy chỉ là bằng chứng trung gian, không thay thế đầu ra chính.
- Estimator, pruning và accounting vượt qua kiểm chứng exact nhỏ cùng kiểm chứng tích hợp.
- Báo cáo phân biệt rõ V_rho, V* của joint kiểm chứng và các giả định còn thiếu đối với native MDM-VGB.
- Có so sánh nhiều seed với tổng chi phí thực, gồm kết quả âm nếu CARS không đem lại lợi ích.
- Kết luận không vượt dữ liệu: không hứa tăng tốc, không gán bảo đảm finite-budget sampling hay mixing-time cho phiên bản chưa thỏa giả định của paper.

Nếu mới chạy được VGB-CARS-rho, hoặc native estimator chưa khả thi/chưa được kiểm chứng, báo cáo đúng phần đã làm và phần thiếu; chưa đánh dấu mục tiêu xấp xỉ V* native hoàn thành. Kết quả không tăng tốc vẫn có thể là kết quả nghiên cứu hợp lệ khi đúng target và đã đo đủ chi phí.

Sau khi hoàn thành phần chính, mới cân nhắc distill label CARS thành neural verifier để giảm chi phí online, online trie update hoặc repair. Mỗi phần mở rộng cần baseline và accounting riêng, bao gồm chi phí tạo label và training.

## 12. Đối chiếu sau Step 14

Step 14 viết entrypoint và đối chiếu tài liệu với source hiện tại; không chạy code/help/test/Modal, không đổi estimator, kernel hoặc số mẫu. Các mục kiểm chứng ở trên vẫn là yêu cầu trước khi kết luận thực nghiệm.

| Thành phần đã viết | Nguồn và mức tương đồng với paper |
| --- | --- |
| QM9, SMILES, 32 token, split 127.190/6.695, vocabulary 35 | Appendix F.1/F.3 của MDM-VGB và config/tokenizer upstream. Asset preparation đã có bằng chứng; corrected token offset là sai khác đã khai báo. |
| Native inference 32 step; label snapshot upstream 128 step | Hai config upstream khác nhau được giữ nhãn riêng. Final denoise có tính NFE là bổ sung chuẩn hóa của dự án. |
| L_f=8, L_b=32, K=8, N=16, lambda=0, chi=1 | Table 7 và `qm9_inference.yaml` là nguồn. Hard total sequence cap, reserved completion, Pass@95 terminal gate và stay của Momentum được ghi riêng trong PAPER_CHECK.md. |
| Standalone CARS-QM9 | Chuyển cơ chế rejection/trie sang backbone/reward QM9; oracle bảo thủ thay grammar functional-group của thí nghiệm CARS gốc. Không tái lập Llama/acrylate/isocyanate hoặc stopping rule 100 valid unique của Appendix H.4. |
| CARS ước lượng giá trị | Discovery rồi frozen-trie fixed-count draws và cache giá trị là thiết kế dự án. Native dùng tử số reweighted trajectory và mẫu số native chưa pruning; MC/neural independent-mask là control cùng target. Chưa có bằng chứng accuracy/feasibility QM9. |
| Protocol đánh giá | Chung checkpoint/calibration, mỗi output yêu cầu được giữ lại, nested cost được tính một lần. Morgan radius 2/2048 theo CARS paper; cap diversity 1.000 molecule là lựa chọn evaluator, không phải output count benchmark. |
| Pilot/final matrix | `configs/benchmark.yaml` vẫn `locked: false`; seed/count và manifests còn null. 128 hoặc 3 seed × 1.000 output không phải setting paper đã xác minh. Final cần full verification hiện tại và pilot cùng scientific contract. |

Entrypoint hỗ trợ 11 task có Modal pair; flags vẫn do runner từng task xác nhận. CARS/MC riêng lẻ không cần train neural; full matrix cần cả upstream-label verifier và matched native-conditional verifier. Native/rho values, toy enumeration và các learned-label targets không được gộp thành một reference.

Các artifact dùng `workspace/` và attempt bất biến; evaluator chọn manifest cụ thể, dùng config snapshot và giữ retry ledger cho end-to-end cost. Verification dùng fixtures offline; kiểm tra production checkpoint và feasibility vẫn cần các task riêng khi được yêu cầu chạy. Importance của joint rho có thư viện/toy checks, chưa được coi là một QM9 benchmark đã chạy; order sensitivity, distillation và editing vẫn là hướng mở rộng.

Mục review còn mở phát hiện khi đối chiếu Step 14: benchmark Step 13 đọc `protocol.json` từ calibration, trong khi writer Step 4 lưu `sampling_protocol.json`. Lỗi nối artifact này được ghi vào checklist implementation.md để xử lý trước preflight; Step 14 giữ nguyên logic Step 13. Chưa có full verification, pilot, benchmark cuối hay kết luận so sánh được bổ sung ở lượt này.
