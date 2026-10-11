# Kế hoạch so sánh CARS và MDM-VGB trên QM9

## 1. Mục tiêu và phạm vi

Xây dựng thí nghiệm trả lời hai câu hỏi:

1. Khi dùng chung mô hình QM9, reward và ngân sách tính toán, CARS và MDM-VGB khác nhau thế nào về tỷ lệ thành công, tính đa dạng và chi phí?
2. Có thể dùng cơ chế loại khối lượng xác suất thất bại của CARS để xấp xỉ giá trị trạng thái, thay thế hoặc hỗ trợ process verifier của MDM-VGB hay không?

CARS là thuật toán lấy mẫu, không phải một dataset QM9 riêng. Benchmark phân tử gốc của CARS dùng các lớp acrylate, chain extender và isocyanate. Dự án này sẽ chuyển cơ chế CARS sang QM9; kết quả của phiên bản chuyển đổi phải được ghi rõ là CARS thích nghi cho QM9.

Tài liệu này chỉ mô tả kế hoạch triển khai và kiểm chứng. Chưa có checkpoint được chạy, chưa có kết quả QM9, và chưa có bằng chứng rằng phương pháp đề xuất vượt baseline. Không thực hiện training, download, sửa thuật toán hoặc chạy benchmark trong bước viết kế hoạch.

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

| Thành phần | Lựa chọn ban đầu |
| --- | --- |
| Kiến trúc | Nguồn model tại resource/MDM-VGB-main/src/tasks/qm9/mdlm_source |
| Dataset | yairschiff/qm9 |
| Tokenizer | yairschiff/qm9-tokenizer |
| Chiều dài | 32 token |
| Checkpoint inference | model_data/qm9/mdlm_qm9/last, tính từ gốc repository upstream |
| Base training | Khởi tạo từ đầu theo cấu hình QM9 upstream, seed 1, 25.000 bước |
| Inference | Cùng checkpoint, tokenizer, precision và quy tắc token hợp lệ giữa các phương pháp |

Checkpoint trên chưa có sẵn trong workspace. Bước prepare assets của upstream tải dataset và tokenizer; base model phải được huấn luyện hoặc được cung cấp bằng một checkpoint có nguồn gốc xác minh được. Việc cung cấp checkpoint không được làm thay đổi âm thầm kiến trúc hoặc vocabulary.

Trong giai đoạn triển khai, kiểm tra vocabulary thực của tokenizer và model; không suy ra nó chỉ từ số 35 trong bảng của paper hoặc số 40 trong cấu hình verifier. Ghi lại hash checkpoint, tokenizer, cấu hình và phiên bản dependency.

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

T là quantile 95% của điểm trên tập base calibration độc lập; giữ nguyên T cho mọi phương pháp trong cùng một giao thức. Báo cáo số lượng calibration, sampler, seed, cách tính quantile và bất định của threshold. Threshold 0.5614854131280362 trong cấu hình hiện tại chỉ được dùng làm cấu hình tham chiếu khi xác minh được nguồn calibration; không mặc định coi nó là threshold tái tạo đúng paper. Không thay calibration bằng percentile QED của dataset.

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

### 5.3. Kiểm chứng khoảng cách tới V* bằng reference có likelihood rõ ràng

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

Chế độ này cung cấp phép kiểm chứng cho xấp xỉ V* trên một reference được xác định đầy đủ. Chỉ mở rộng tới QM9 nếu pilot cho thấy variance và chi phí chấp nhận được. Nếu không, báo cáo giới hạn đó và giữ kết quả native MDM-VGB là đánh giá thực nghiệm dùng V_rho. Dùng giá trị đã hiệu chỉnh cùng logits native vẫn không tạo ra bảo đảm lấy mẫu chính xác từ joint pi_rho, vì transition conditionals cũng cần khớp reference.

### 5.4. Cache và tính ổn định

Khóa cache gồm toàn bộ token của z, checkpoint, tokenizer, reward/threshold, policy rho, quy tắc token đặc biệt và cấu hình estimator. Không coi hai trạng thái có cùng prefix nhưng token tương lai khác nhau là một trạng thái.

Đóng băng giá trị đã hoàn tất của một trạng thái trong mỗi run, để cùng một trạng thái không thay đổi giá trị do một truy vấn mới. Nếu thử online update, coi đó là ablation riêng và không kế thừa stationary-law guarantee của kernel cố định. Giới hạn bộ nhớ bằng eviction được ghi nhận; eviction làm phát sinh lại chi phí và không cho phép dùng cache thiếu ngữ cảnh.

## 6. Tích hợp và bố trí phần triển khai dự kiến

| Khu vực dự kiến | Trách nhiệm |
| --- | --- |
| src/cars_vgb/models | Kiến trúc độc lập, adapter model/tokenizer và thống kê forward calls |
| src/cars_vgb/qm9 | Split, canonicalization, terminal reward, calibration và rollout |
| src/cars_vgb/sampling | Native sampling, completion/reference, Base, BoN, rejection, VGR, VGB và Momentum |
| src/cars_vgb/cars | Trie, oracle, proposal và estimator giá trị |
| src/cars_vgb/verifiers | Adapter CARS, Monte Carlo, neural verifier và importance mode qua cùng giao diện |
| src/cars_vgb/training | Training base model và neural verifier |
| src/cars_vgb/experiments | Run, seed, budget, panel giá trị và điều phối benchmark |
| src/cars_vgb/evaluation | Đọc artifact, tính metrics, tổng hợp bất định và vẽ đồ thị |
| src/cars_vgb/execution/modal_runtime.py | Cấu hình image, dependency, Volume và tài nguyên Modal dùng chung |
| modal | Mỗi task một file Python và một script shell gọi modal run |
| configs | Reference mode, reward, estimator, sampler và budget |
| tests | Các kiểm chứng có ý nghĩa về xác suất, pruning, estimator và tích hợp |
| results/raw và results/summary | Artifact từng run và bảng/đồ thị tái tạo được |

Phần structure bổ sung cho Modal như sau. README.md trong modal đã tồn tại; các file Python, shell và cấu hình trong cây này sẽ được tạo ở bước IMPLEMENT. Các module thuật toán còn lại giữ bố trí ở bảng trên.

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
│   └── modal.yaml                      # GPU/CPU, memory, timeout, concurrency và Volume
└── src/
    └── cars_vgb/
        └── execution/
            ├── __init__.py             # Khai báo package điều phối thực thi
            └── modal_runtime.py        # Image, dependency và Volume dùng chung
```

Modal là môi trường chạy các task của dự án ngay từ đầu. Các kiểm chứng nhỏ vẫn có thể chạy local, và cũng có task verify để chạy chúng trong image Modal. Logic khoa học nằm trong package src/cars_vgb; các file modal chỉ cấu hình tài nguyên, nhận tham số và gọi logic tương ứng.

Mỗi task có cặp modal/<task>.py và modal/run_<task>.sh. Script gọi modal run modal/<task>.py::main, chuyển tiếp tham số, giữ exit code và xác định project root để không phụ thuộc thư mục gọi lệnh. Đây là quy ước cho bước triển khai sau, chưa tạo file thực thi trong STEP 4. CLI chính thức hỗ trợ tham chiếu file::entrypoint: [Modal run](https://modal.com/docs/cli/latest/run).

Danh sách task được chốt trong [modal/README.md](modal/README.md): prepare_assets, train_base, calibrate, collect_rollouts, train_verifier, prepare_value_panel, generate, check_values, benchmark, evaluate và verify. File generate nhận method/config để chạy từng phương pháp; benchmark điều phối các run theo ma trận đã khóa. Cấu hình tài nguyên dùng chung nằm tại configs/modal.yaml, tách khỏi siêu tham số thuật toán. Không tạo modal/__init__.py để tránh che package SDK modal khi chạy từ project root.

Dataset, tokenizer, checkpoint, snapshot trie và kết quả cần giữ giữa các task được lưu trên một Modal Volume, mount tại /workspace với các thư mục data, model_data, cache/tries và results. Các task dùng ID và manifest để tham chiếu đúng artifact. Checkpoint model được tái sử dụng; thay verifier không kích hoạt training backbone lại. Snapshot trie chỉ được nạp nếu metadata khớp đầy đủ khóa cache ở mục 5.4. Triển khai commit sau khi ghi và reload trước khi đọc artifact mới từ task khác; mỗi run/attempt có đường dẫn riêng, tránh nhiều worker ghi cùng file. Cơ chế lưu trữ theo [Modal Volumes](https://modal.com/docs/guide/volumes).

Image chỉ đưa vào các file cần chạy từ package độc lập, configs, entrypoint Modal, metadata dependency/license và tests cùng fixtures cho task verify. Không mount hoặc upload resource. Task evaluate lưu báo cáo vào Volume và hỗ trợ tải các artifact được chọn về results ở máy local; không tải lại toàn bộ dataset/checkpoint sau mỗi run.

Adapter verifier CARS sẽ hỗ trợ chấm theo batch và trả giá trị theo đúng thứ tự trạng thái đầu vào. Dùng lại điểm tích hợp của QM9Verifier. Trạng thái hoàn chỉnh được neo bằng terminal reward chính xác; các thống kê độ bất định, cache và chi phí được lưu riêng.

Không sửa backbone khi thay verifier. Trong so sánh neural–MC–CARS, giữ nguyên graph, cách shortlist, seed policy và chính sách terminal; chỉ thay nguồn giá trị. Huấn luyện thêm một neural verifier với label theo q_rho để có baseline cùng target; giữ verifier học từ native diffusion rollout thành cấu hình riêng để phân biệt ảnh hưởng của target và estimator.

## 7. Baseline và ma trận so sánh

| Nhóm | Phương pháp | Câu hỏi |
| --- | --- | --- |
| Native QM9 | Base, BoN | Chất lượng unguided và baseline outcome-only |
| Native QM9 | VGR, VGB, VGB-Momentum với learned verifier upstream | Mốc tham chiếu theo giao thức MDM-VGB |
| Cùng q_rho | Base-rho, rejection sampling, ARS, CARS | Giá trị của adaptive pruning trên cùng luật sinh |
| Cùng graph VGB | Neural-rho, MC-rho, CARS-rho verifier | CARS có giúp đánh giá trạng thái và cải thiện generation hay không |
| Reference kiểm chứng | Exact values, MC và CARS có importance correction | Estimator có khớp V* của joint đã chỉ định hay không |

Không gộp native Base và Base-rho thành một baseline, dù dùng chung trọng số. Không dùng Llama/Qwen từ CARS gốc làm đối chứng chính với QM9 MDLM vì như vậy thay đổi cả model và sampler.

Thiết lập ban đầu bám cấu hình upstream: chiều dài 32, L_f = 8, L_b = 32, K = 8, singleton updates, lambda = 0 và chi = 1. Kiểm tra sensitivity với lambda và reveal order sau khi đường chạy cơ bản đúng. Giữ các default được kiểm tra từ code tách biệt với tuyên bố tái tạo đúng mọi kết quả paper.

Chạy pilot một seed trên khoảng 128 output mỗi cấu hình và một panel trạng thái nhỏ. Khảo sát N thuộc 1, 2, 4, 8, 16; discovery budget và số draw thuộc một grid nhỏ đã công bố. Sau profiling, khóa subset khả thi trước khi nhìn tập đánh giá cuối. Mục tiêu benchmark chính là ít nhất ba seed độc lập và 1.000 output mỗi cấu hình được chọn; tăng lên năm seed khi tài nguyên cho phép. Nếu cần giảm quy mô, ghi nhận trước và báo cáo độ bất định tương ứng.

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

Ground truth chính xác chỉ đến từ enumeration trên không gian nhỏ hoặc trạng thái ít mask có reference joint tính được. Monte Carlo nhiều mẫu trên QM9 là giá trị tham chiếu có bất định, không được gọi là ground truth chính xác. So sánh cùng target trước khi báo cáo sai số; không đo lỗi của V_rho bằng một estimator thuộc native rollout khác rồi bỏ qua target mismatch.

## 9. Các bước triển khai theo thứ tự

1. **Chốt giao thức và môi trường:** xác minh dependency, assets, model/tokenizer và các quy tắc trạng thái; lưu cấu hình và manifest dữ liệu.
2. **Chuẩn bị backbone:** lấy checkpoint có provenance hoặc huấn luyện đúng cấu hình; đóng băng và kiểm tra base generation.
3. **Xây reward và calibration:** hoàn thiện terminal checks, tạo calibration độc lập, khóa threshold và panel đánh giá.
4. **Tái tạo baseline:** native Base/BoN/VGR/VGB/Momentum, rollout và learned verifier; ghi rõ sai khác với paper.
5. **Xác định q_rho và MC-rho:** tạo đường hoàn tất cố định, đo sự phụ thuộc reveal order và dựng baseline giá trị không pruning.
6. **Xây trie và estimator:** bắt đầu với leaf rejection, xác minh khối lượng, thêm grammar pruning an toàn, rồi frozen-trie estimation.
7. **Kiểm chứng V* trên reference rõ ràng:** enumeration, importance correction và kiểm tra support; mở rộng QM9 khi chi phí cho phép.
8. **Tích hợp verifier:** dùng cùng VGB graph, accounting và policy cho neural-rho, MC-rho và CARS-rho; giữ terminal anchoring.
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
- Có baseline QM9, MC-rho, CARS thích nghi và VGB dùng các verifier tương ứng chạy qua cùng evaluator.
- Estimator, pruning và accounting vượt qua kiểm chứng exact nhỏ cùng kiểm chứng tích hợp.
- Báo cáo phân biệt rõ V_rho, V* của joint kiểm chứng và các giả định còn thiếu đối với native MDM-VGB.
- Có so sánh nhiều seed với tổng chi phí thực, gồm kết quả âm nếu CARS không đem lại lợi ích.
- Kết luận không vượt dữ liệu: không hứa tăng tốc, không gán bảo đảm finite-budget sampling hay mixing-time cho phiên bản chưa thỏa giả định của paper.

Sau khi hoàn thành phần chính, mới cân nhắc distill label CARS thành neural verifier để giảm chi phí online, online trie update hoặc repair. Mỗi phần mở rộng cần baseline và accounting riêng, bao gồm chi phí tạo label và training.
