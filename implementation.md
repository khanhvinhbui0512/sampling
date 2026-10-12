# Thứ tự triển khai và tổ chức mã nguồn

Tài liệu này xác định cấu trúc project và thứ tự hoàn thiện từng tệp cho project so sánh CARS, MDM-VGB và CARS + MDM-VGB trên benchmark Molecular Generation (SMILES) của CARS. Mỗi mục được đánh số trong giai đoạn 1–12 tương ứng với một tệp cần triển khai, kèm chức năng và tiêu chí hoàn thành.

## Quy tắc thực hiện

- Đi theo thứ tự 1.1, 1.2, ... trong từng giai đoạn rồi chuyển sang giai đoạn tiếp theo. Hoàn thiện chức năng và tiêu chí của một tệp trước khi viết tệp kế tiếp; không để TODO, thuật toán giả hoặc stub thay cho chức năng đã mô tả.
- Một tệp thư viện được coi là hoàn thành khi thực hiện đầy đủ hợp đồng của nó, dùng các thành phần đã có và có thể kiểm tra bằng đầu vào nhỏ. Không cần chờ một checkpoint huấn luyện lớn để hoàn thiện thuật toán.
- Tệp cấu hình có thể chỉ định đường dẫn artifact sẽ được task tạo sau. Phải kiểm tra artifact tại thời điểm chạy task; thiếu dependency phải báo lỗi rõ ràng.
- Mỗi task T01–T16 có một module local và một wrapper cùng tên trong modal/. Wrapper gọi lại task dùng chung, nhận cùng tham số nghiệp vụ; thuật toán và bộ đánh giá nằm trong src/.
- Project chỉ xử lý benchmark Molecular Generation (SMILES); ba nhóm hóa học là các điều kiện của cùng benchmark. Các module nằm trực tiếp trong src/, các task nằm trong src/tasks/. Dùng import src.* và chạy local bằng python src/tasks/<task>.py từ project root sau khi cài package editable; chạy cloud bằng python modal/<task>.py. Wrapper Modal gọi cùng logic trong src.tasks.<task>, dùng App/Image/Volume và app.run().
- Local dùng Miniconda để tạo môi trường Python 3.12, rồi cài dependencies bằng pip. Modal dùng tag nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 với Python 3.12 và pip; dependencies lấy từ pyproject.toml. requirements.lock là tùy chọn; báo cáo môi trường ghi phiên bản đã cài.
- Các tệp trong data/, checkpoints/, outputs/ và reports/ là artifact được task sinh ra. Không viết thủ công trọng số, dữ liệu đã xử lý hoặc kết quả thực nghiệm.
- resource/ là nguồn tham khảo. Nếu cần dùng lại nội dung, sao chép có chọn lọc hoặc viết mới trong project, ghi nguồn và giấy phép. Mã runtime không import, thêm vào sys.path, dùng symlink hoặc phụ thuộc vào tệp trong resource/.
- Chạy thử nhỏ để kiểm tra pipeline trước. Huấn luyện đầy đủ và đánh giá test chỉ chạy sau các kiểm tra thuật toán, tích hợp và bộ đếm chi phí.

## Phần giữ nguyên và phần thay đổi so với paper

| Thành phần | Thiết kế được triển khai | Quan hệ với paper |
|---|---|---|
| Benchmark | Ba nhóm acrylates, chain extenders, isocyanates; prompt, grammar và 32/11/11 ví dụ gốc | Giữ benchmark CARS |
| Corpus huấn luyện | ChEMBL cố định phiên bản, đề xuất ChEMBL 36; split theo phân tử | Phần bổ sung; CARS dùng LLM pretrained, MDM-VGB dùng QM9 |
| Mô hình chung | MDLM SMILES có điều kiện nhóm, dùng cùng checkpoint cho ba phương pháp | Thích nghi từ thiết lập MDLM của MDM-VGB |
| Reward chính | Grammar AND RDKit-valid AND membership theo SMARTS benchmark | Thay đổi có chủ đích so với grammar acceptance của mã CARS và QED Pass@95 của QM9 |
| CARS trên MDLM | Hoàn thiện các vị trí che theo thứ tự cố định, tạo chính sách q_z | Adapter mới; không phải thiết lập autoregressive gốc |
| MDM-VGB | Giữ unmask/remask, shortlisting, terminal anchoring; thay phần đặc thù tác vụ | Giữ cơ chế, thích nghi dữ liệu/tokenizer/reward |
| Hybrid | Thay learned value bằng ước lượng CARS | Đề xuất nghiên cứu mới |
| Tái lập CARS | Llama-3.1-8B-Instruct, prompt gốc, grammar-only acceptance, cấu hình bài báo | Suite riêng để đối chiếu CARS gốc |

Chỉ gọi bộ ước lượng hybrid là xấp xỉ V* của MDM-VGB khi nêu rõ điều kiện về phân phối tham chiếu. Tính không chệch đối với chính sách q_z không chứng minh q_z bằng phân phối conditional lý tưởng của MDM-VGB. Nếu chưa có chứng minh tương ứng, báo cáo V_q, sai số Monte Carlo theo q_z và sự khác biệt với rollout tham chiếu.

## Cấu trúc toàn project

Tên tệp và vai trò chi tiết được giải thích trong thứ tự triển khai bên dưới. T01–T16 là mã tham chiếu của các cặp tệp script local và wrapper Modal.

```text
smiles/
├── README.md
├── PLAN.md
├── implementation.md
├── pyproject.toml
├── requirements.lock                # Tùy chọn; không bắt buộc để chạy Modal
├── .gitignore
├── THIRD_PARTY_NOTICES.md
├── resource/                         # Nguồn tham khảo hiện có; không đóng gói runtime
│   ├── 2510.01902v2.pdf
│   ├── 2606.28301v1.pdf
│   ├── CARS-code/                     # Nội dung tham khảo gốc
│   └── MDM-VGB/                       # Nội dung tham khảo gốc
├── assets/
│   └── benchmark/
│       ├── source.json
│       ├── smarts.json
│       ├── acrylates.prompt.txt
│       ├── acrylates.examples.smi
│       ├── acrylates.lark
│       ├── chain_extenders.prompt.txt
│       ├── chain_extenders.examples.smi
│       ├── chain_extenders.lark
│       ├── isocyanates.prompt.txt
│       ├── isocyanates.examples.smi
│       └── isocyanates.lark
├── configs/
│   ├── common.yaml
│   ├── data.yaml
│   ├── models.yaml
│   ├── mdlm.yaml
│   ├── rollouts.yaml
│   ├── verifier.yaml
│   ├── base.yaml
│   ├── cars.yaml
│   ├── mdm_vgb.yaml
│   ├── hybrid.yaml
│   ├── replication.yaml
│   ├── verification.yaml
│   ├── value_eval.yaml
│   ├── evaluation.yaml
│   ├── comparison.yaml
│   ├── report.yaml
│   └── modal.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── artifacts.py
│   ├── tracking.py
│   ├── data.py
│   ├── tokenizer.py
│   ├── mdlm.py
│   ├── autoregressive.py
│   ├── completion.py
│   ├── grammar.py
│   ├── reward.py
│   ├── verifier.py
│   ├── trie.py
│   ├── cars.py
│   ├── mdm_vgb.py
│   ├── values.py
│   ├── hybrid.py
│   ├── reference.py
│   ├── metrics.py
│   ├── retrosynthesis.py
│   ├── reporting.py
│   └── tasks/
│       ├── __init__.py
│       ├── common.py
│       ├── prepare_benchmark.py
│       ├── prepare_corpus.py
│       ├── build_tokenizer.py
│       ├── fetch_models.py
│       ├── train_mdlm.py
│       ├── build_rollouts.py
│       ├── train_verifier.py
│       ├── generate_base.py
│       ├── generate_cars.py
│       ├── generate_mdm_vgb.py
│       ├── generate_hybrid.py
│       ├── verify.py
│       ├── evaluate_value.py
│       ├── evaluate.py
│       ├── compare.py
│       └── report.py
├── modal/                            # Entry point và hạ tầng Modal; không có __init__.py
│   ├── common.py
│   ├── prepare_benchmark.py
│   ├── prepare_corpus.py
│   ├── build_tokenizer.py
│   ├── fetch_models.py
│   ├── train_mdlm.py
│   ├── build_rollouts.py
│   ├── train_verifier.py
│   ├── generate_base.py
│   ├── generate_cars.py
│   ├── generate_mdm_vgb.py
│   ├── generate_hybrid.py
│   ├── verify.py
│   ├── evaluate_value.py
│   ├── evaluate.py
│   ├── compare.py
│   └── report.py
├── tests/
│   ├── conftest.py
│   ├── test_resource_isolation.py
│   ├── test_data_reward.py
│   ├── test_cars.py
│   ├── test_values.py
│   ├── test_integration.py
│   └── test_metrics.py
├── data/
│   ├── benchmark/
│   │   ├── examples.parquet
│   │   └── manifest.json
│   ├── corpus/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   ├── test.parquet
│   │   └── manifest.json
│   ├── tokenizer/
│   │   ├── vocab.json
│   │   └── metadata.json
│   ├── tokenized/<split>/
│   │   ├── shard_<id>.pt
│   │   └── manifest.json
│   ├── rollouts/<run_id>/
│   │   ├── shard_<id>.pt
│   │   └── manifest.json
│   └── value_eval/<run_id>/
│       ├── states.pt
│       └── reference_values.parquet
├── checkpoints/
│   ├── pretrained/<model_id>/<revision>/  # Snapshot bên thứ ba theo manifest
│   └── <mdlm_or_verifier>/<run_id>/<step>/
│       ├── model.safetensors
│       ├── training_state.pt
│       └── metadata.json
├── outputs/<suite_id>/<method>/<group>/seed_<seed>/
│   ├── resolved_config.yaml
│   ├── manifest.json
│   ├── attempts.jsonl
│   ├── samples.jsonl
│   ├── events.jsonl
│   └── metrics.json
└── reports/<suite_id>/
    ├── summary.csv
    ├── value_accuracy.csv
    ├── quality_cost.png
    ├── value_calibration.png
    └── report.md
```

Các tên chứa <...> là mẫu đường dẫn, được thay bằng ID khi chạy. requirements.lock là artifact tùy chọn để ghi lại dependencies; image Modal cài trực tiếp từ pyproject.toml. Các tệp trong data/, checkpoints/, outputs/, reports/ là artifact tự sinh. Nội dung các repository trong resource/ và snapshot pretrained được coi là hiện vật bên ngoài; không liệt kê từng tệp nội bộ của chúng như mã cần triển khai.

So với STEP 4, bổ sung tests/test_metrics.py để kiểm tra trực tiếp công thức và mẫu số đánh giá. PLAN.md đã có; implementation.md là tài liệu của bước hiện tại.

## 1) Package và quản lý tệp

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 1.1 | pyproject.toml | Định nghĩa package theo cấu trúc src/, dependencies cho mô hình/dữ liệu/chemistry, extras modal và dev. Giải quyết cài đặt không nhất quán giữa local và cloud. Đóng gói package src và src.* từ project root, với src/__init__.py; không cần README để bootstrap package, không cài repository trong resource/. |
| 1.2 | .gitignore | Quy định loại cache, môi trường ảo, dữ liệu lớn, checkpoint và kết quả khỏi Git. Giữ assets nhỏ, cấu hình và tài liệu trong version control. |
| 1.3 | THIRD_PARTY_NOTICES.md | Ghi nguồn, commit, giấy phép của benchmark và mọi đoạn mã thực sự sao chép. Phân biệt viết lại thuật toán theo paper với dùng lại mã. Hoàn thành danh sách nguồn cho phạm vi triển khai đã định. |
| 1.4 | src/__init__.py | Khai báo package src và phiên bản; import nhẹ, không tải model, đọc dữ liệu hay khởi tạo Modal. Giải quyết tác dụng phụ khi import. |
| 1.5 | src/tasks/__init__.py | Khai báo namespace cho các module CLI; không tự import tất cả task. Cho phép chạy từng module độc lập khi task khác chưa được triển khai. |

## 2) Hợp đồng dữ liệu, cấu hình và ghi kết quả

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 2.1 | src/schemas.py | Định nghĩa state có token IDs, mask, nhóm và quy tắc EOS/PAD; giao diện model/value provider; record attempt/sample/run. Phân biệt root attempt, prefix abort, leaf visit và mẫu được chọn để tránh mẫu số mơ hồ. |
| 2.2 | configs/common.yaml | Chốt tên ba nhóm, seed, phiên bản reward, protocol và đường dẫn tương đối. Suite chính đặt mục tiêu 100 phân tử duy nhất có tau=1, loại ví dụ trong prompt, tối đa 1.000 root attempts cho mỗi nhóm/seed. Phân biệt suite chính, replication và smoke; khai báo nơi lưu artifact local/Modal. |
| 2.3 | src/config.py | Đọc YAML, hợp nhất common với cấu hình task, áp dụng CLI overrides và kiểm tra kiểu/ràng buộc. Giải quyết tham số sai hoặc cấu hình ngầm. Xuất cấu hình đã resolve; không import module task chưa tồn tại. |
| 2.4 | src/artifacts.py | Ánh xạ đường dẫn theo artifact root, lưu/nạp manifest, checksum, ghi atomically và kiểm tra dependency. Resume phải xác nhận cùng config/model/tokenizer/reward và bảo toàn RNG, bộ đếm, trạng thái sampler cần thiết; không nối kết quả từ run khác. |
| 2.5 | src/tracking.py | Đếm mọi MDLM/AR forward, learned-verifier call, CARS attempt, cache hit và chi phí CPU/GPU. Phân biệt thời gian compute với thời gian end-to-end của Modal. Các completion trong CARS-value được tính vào NFE của model. |
| 2.6 | src/tasks/common.py | Dùng chung CLI parsing, tạo run context, seed/RNG, config và artifact root. Nhận callable của task thay vì import registry tương lai. Hỗ trợ config, run ID, resume và smoke; group/seed chỉ áp dụng cho task phù hợp. |
| 2.7 | tests/conftest.py | Tạo phân phối hữu hạn có thể liệt kê, model/value provider đồ chơi và fixture phân tử nhỏ. Giải quyết việc kiểm tra xác suất và state transitions mà không cần checkpoint lớn. |

**Phụ thuộc:** chỉ dùng package/môi trường ở giai đoạn 1. Sau giai đoạn này, các task sau có chung định dạng input/output và cách ghi chi phí.

## 3) Hạ tầng Modal và task kiểm tra

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 3.1 | configs/modal.yaml | Gán CPU/GPU, RAM, timeout, image profile và Volume cho từng task. Task điều phối dùng tài nguyên riêng; worker huấn luyện/sinh mẫu được cấu hình theo nhu cầu. Không hard-code loại GPU trong thuật toán. |
| 3.2 | modal/common.py | Tạo image bằng Image.from_registry với tag CUDA đã chốt và Python 3.12; dùng Image.pip_install với dependencies đọc từ pyproject.toml, rồi add_local_file cho các tệp project được cho phép. Quản lý Volume.from_name, model-access Secret khi cần, commit/reload, tài nguyên worker và CLI cho python modal/<task>.py. Mỗi wrapper dùng @app.function và if __name__ == "__main__" với enable_output()/app.run(). Dùng path đến helper, không import modal.common vì tên modal thuộc SDK. Helper không chứa thuật toán CARS/MDM-VGB. |
| 3.3 | tests/test_resource_isolation.py | Kiểm tra imports và đường dẫn runtime; package hoạt động với src/configs/assets mà không có resource/. Kiểm tra thư mục modal/ không che SDK và không được phát hiện như package của project. |
| 3.4 | configs/verification.yaml | Định nghĩa các phase environment, benchmark, corpus, cars, values, integration, metrics và all. Mỗi phase chỉ định test và artifact bắt buộc; phase thiếu artifact phải thất bại rõ ràng. |
| 3.5 | src/tasks/verify.py — T12 | Triển khai engine chạy phase được chọn và xuất báo cáo. Engine nhận test paths từ cấu hình, không import trực tiếp các module tương lai. Có thể hoàn thiện ngay bằng phase environment; chức năng chạy mọi phase là chung. |
| 3.6 | modal/verify.py — T12 | Chạy cùng engine kiểm tra trong image Modal; nhận cùng phase/config và lưu báo cáo. Giải quyết khác biệt môi trường trước các job GPU lớn. |

Tất cả wrapper Modal tiếp theo được hoàn thiện sau task local tương ứng. Wrapper phải parse/forward đúng tham số, cấu hình tài nguyên, lưu artifact và trả trạng thái; chạy thử nhỏ khi input của task đã sẵn sàng.

## 4) Benchmark, tokenizer, grammar và reward

Chuẩn bị asset từ kho tác giả CARS tại commit f46607f983d89e4ca4f40527cd231ab04adc6ed7. Đây là bước sao chép dữ liệu tham chiếu vào assets/, không import mã của repository tham khảo.

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 4.1 | assets/benchmark/acrylates.prompt.txt | Lưu nguyên prompt gốc để tái lập few-shot CARS, giữ đúng nội dung nguồn. |
| 4.2 | assets/benchmark/acrylates.examples.smi | Trích đủ 32 ví dụ, lưu từng SMILES để chuẩn hóa và kiểm tra novelty. Có thể truy vết từng dòng về prompt gốc. |
| 4.3 | assets/benchmark/acrylates.lark | Lưu grammar gốc của acrylates; bảo toàn nội dung và ghi checksum. |
| 4.4 | assets/benchmark/chain_extenders.prompt.txt | Lưu nguyên prompt nhóm chain extenders. |
| 4.5 | assets/benchmark/chain_extenders.examples.smi | Trích đủ 11 ví dụ chain extenders, phục vụ nhận diện trùng phân tử. |
| 4.6 | assets/benchmark/chain_extenders.lark | Lưu grammar gốc nhóm chain extenders. |
| 4.7 | assets/benchmark/isocyanates.prompt.txt | Lưu nguyên prompt nhóm isocyanates. |
| 4.8 | assets/benchmark/isocyanates.examples.smi | Trích đủ 11 ví dụ isocyanates. |
| 4.9 | assets/benchmark/isocyanates.lark | Lưu grammar gốc nhóm isocyanates. |
| 4.10 | assets/benchmark/smarts.json | Cố định pattern gốc: acrylate C=CC(=O)O*, isocyanate [*]N=C=O, chain extender CO/OC/N. Ghi rõ chain matcher không bảo đảm tính hai chức; một phân tử có thể khớp nhiều nhóm. |
| 4.11 | assets/benchmark/source.json | Ghi URL, commit, file gốc, số ví dụ và checksum của các asset đã có. Giải quyết việc dùng nhầm prompt rút gọn hoặc phiên bản grammar khác. |
| 4.12 | src/data.py | Đọc/ghi dữ liệu, sanitize/canonicalize, định danh phân tử, chống trùng và split theo phân tử. Nhận bộ phân loại nhóm từ caller để tránh circular import với reward.py. Cùng một phân tử không được rơi vào nhiều split dù có nhiều SMILES hoặc nhiều nhãn. |
| 4.13 | src/tokenizer.py | Định nghĩa tokenize/detokenize SMILES, xây vocabulary và MASK/EOS/PAD, kiểm tra round-trip và token không hỗ trợ. Giao diện được khai báo trong schemas; chưa cần corpus lớn để kiểm tra trên fixture. |
| 4.14 | src/grammar.py | Compile grammar và cung cấp kiểm tra chuỗi/prefix/token continuation. Hỗ trợ tokenizer SMILES và tokenizer AR qua cùng giao diện. Chỉ chứng nhận prefix thất bại khi có cơ sở chắc chắn; trạng thái chưa biết không bị coi là thất bại. |
| 4.15 | src/reward.py | Là nguồn định nghĩa reward chính và grammar-only replication. Dùng cùng logic cho label, terminal value và terminal_accept. Tách grammar, RDKit validity, membership; không kết thúc khi còn MASK trước EOS, không che lỗi bằng bỏ token. |
| 4.16 | tests/test_data_reward.py | Kiểm tra canonical identity, split/augmentation, tokenizer special tokens và các trường hợp grammar-valid nhưng không đạt RDKit/membership. Kiểm tra một molecule có nhiều nhãn vẫn không rò rỉ split. Các kiểm tra fixture và kiểm tra corpus thực được chọn theo phase T12. |
| 4.17 | configs/data.yaml | Chốt nguồn benchmark/ChEMBL, phiên bản, seed split, cách chuẩn hóa, lọc và augmentation. Đề xuất split 80/10/10; loại ví dụ benchmark trước khi chia. Tokenizer và báo cáo độ dài có artifact riêng. |
| 4.18 | src/tasks/prepare_benchmark.py — T01 | Xác minh asset, chuẩn hóa các ví dụ, đối chiếu counts và ghi examples.parquet/manifest.json. Báo riêng mismatch grammar/RDKit/SMARTS; không tự sửa dữ liệu gốc để làm kiểm tra đạt. |
| 4.19 | modal/prepare_benchmark.py — T01 | Chạy T01 trên Modal với asset đóng gói của project và ghi artifact vào Volume. |

## 5) Corpus huấn luyện và token hóa

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 5.1 | src/tasks/prepare_corpus.py — T02 | Tải bản ChEMBL đã cố định, sanitize/deduplicate, loại ví dụ benchmark, gán nhãn bằng checker chung rồi split theo molecule. Chỉ augmentation sau split. Ghi số phân tử mỗi nhóm, overlap nhóm và các nguyên nhân bị lọc. |
| 5.2 | modal/prepare_corpus.py — T02 | Chạy preprocessing trên tài nguyên CPU/RAM phù hợp và lưu các split bền vững. |
| 5.3 | src/tasks/build_tokenizer.py — T03 | Xây vocabulary theo quy tắc đã chốt từ train, token hóa các split và ghi shards. Không dùng test để chọn vocabulary hoặc tune; báo unknown tokens và tỷ lệ bị loại do độ dài trên mọi split. |
| 5.4 | modal/build_tokenizer.py — T03 | Chạy T03 với cùng dữ liệu và config, lưu tokenizer/shards vào Volume. |

**Artifact cần có:** corpus manifest, ba split, tokenizer metadata và manifest của dữ liệu token hóa. Đây là corpus bổ sung cho MDLM, không phải dữ liệu huấn luyện gốc của paper CARS.

## 6) Mô hình nền, completion và huấn luyện MDLM

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 6.1 | configs/models.yaml | Chốt model ID/revision của Llama và model retrosynthesis, tokenizer đi kèm và expected manifest. Đánh dấu pretrained cần cho replication/evaluation; MDLM chính được huấn luyện từ đầu. |
| 6.2 | src/autoregressive.py | Adapter load Llama, prompt formatting và next-token probabilities cho CARS replication. Giữ temperature 1, top-p 1, top-k 0; ghi revision/tokenizer và dtype. Không thay mô hình MDLM của hybrid bằng Llama. |
| 6.3 | src/tasks/fetch_models.py — T04 | Download/cache snapshot theo revision và kiểm tra tệp cần thiết. Không tự đổi sang model khác khi thiếu quyền truy cập; chỉ tải model được cấu hình. |
| 6.4 | modal/fetch_models.py — T04 | Tải snapshot vào Volume, dùng Secret truy cập model khi cần, kiểm tra tải hoàn tất trước task sau. |
| 6.5 | src/mdlm.py | Triển khai kiến trúc MDLM, time/mask conditioning, class conditioning, masked objective và batch token probabilities. Xử lý PAD đúng và chỉ dự đoán token hợp lệ của vocabulary; kiểm tra loss/gradient trên minibatch nhỏ. |
| 6.6 | src/completion.py | Hoàn thiện q_z theo thứ tự trái sang phải và hoàn thiện theo policy tham chiếu đã định. Giữ token đã hiện, xử lý EOS/PAD và trạng thái có known token sau vị trí che. Có thể truy vết xác suất nhánh cho trie; dùng đầy đủ vocabulary. |
| 6.7 | configs/mdlm.yaml | Điểm khởi đầu: 12 lớp, hidden 768, 12 heads, dropout 0,1, batch toàn cục 256, LR 3e-4, warmup 1.000, 25.000 bước, min LR 3e-6. Độ dài ban đầu 256 được chốt lại từ train/validation; chế độ smoke có kích thước/ngân sách nhỏ, metadata riêng. |
| 6.8 | src/tasks/train_mdlm.py — T05 | Huấn luyện trên train, chọn checkpoint theo validation, lưu weights/optimizer/scheduler/RNG và resume. Test không dùng chọn checkpoint. Resume xác nhận tokenizer, nhóm và corpus hashes. |
| 6.9 | modal/train_mdlm.py — T05 | Chạy cùng trainer trên GPU cấu hình, checkpoint định kỳ vào Volume và tiếp tục từ artifact hợp lệ. |

## 7) Rollout và learned verifier

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 7.1 | configs/rollouts.yaml | Khởi đầu 128 bước và 5 snapshot/rollout như config QM9; chốt policy, nhóm, seed và sampling depth cho độ dài mới. Ghi checkpoint nguồn và phiên bản reward. |
| 7.2 | src/tasks/build_rollouts.py — T06 | Sinh completion, lưu trạng thái che và nhãn reward terminal. Tất cả snapshot cùng rollout phải cùng partition; các completion cùng molecule không rơi vào train/validation khác nhau. Với đầu ra invalid, dùng identity chuỗi rõ ràng để chống trùng. |
| 7.3 | modal/build_rollouts.py — T06 | Chạy rollout theo batch trên GPU và ghi shard riêng, không ghi đồng thời một shard từ nhiều worker. |
| 7.4 | src/verifier.py | Transformer nhỏ nhận state + nhóm và trả value scalar qua sigmoid. Cung cấp batch inference, loss MSE và save/load tương thích state/tokenizer. Hoàn thiện bằng dữ liệu rollout nhỏ trước khi training lớn. |
| 7.5 | configs/verifier.yaml | Điểm khởi đầu: 4 lớp, hidden 256, 4 heads, dropout 0,1, sigmoid, LR 1e-4, batch toàn cục 2.048, tối đa 200 epoch. Chọn checkpoint theo validation và ghi reward/checkpoint rollout. |
| 7.6 | src/tasks/train_verifier.py — T07 | Regression reward trên snapshot, validation theo partition đã khóa, lưu training state và kiểm tra calibration. Không dùng trạng thái test của T13 để chọn checkpoint. |
| 7.7 | modal/train_verifier.py — T07 | Wrapper training verifier với cùng trainer và cơ chế checkpoint/resume. |

## 8) CARS hoàn chỉnh trước khi tích hợp

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 8.1 | src/trie.py | Lưu prefix, xác suất nhánh và failed cones/leaves; cập nhật mass theo log-space, cache theo model/tokenizer/state/group/policy. Xử lý mass bằng 0, overlap prefix và cập nhật lặp mà không đếm khối xác suất hai lần. |
| 8.2 | src/cars.py | Triển khai CARS trên completion interface, parser feedback, root constraint và cập nhật mass. Prefix pruning phải sound; lá RDKit/SMARTS thất bại chỉ loại chính lá đó nếu chưa chứng minh cone thất bại. Trả cả attempt thành công/thất bại và mass cần cho value estimator. |
| 8.3 | configs/cars.yaml | CARS trên MDLM chung: temperature 1, full vocabulary, learn_level 3/constrain_first tương đương thiết kế gốc, reward chính và budget toàn cục. Chốt policy q_z và quy tắc lưu trie. |
| 8.4 | configs/replication.yaml | CARS AR/Llama: prompt gốc, revision 0e9e39f, BF16, temperature 1/top-p 1/top-k 0, max_new_tokens 256; grammar-only acceptance, 100 mẫu duy nhất, cap 1.000 lượt thử, ba seed. Ghi khác biệt với default 512/2.000 của script tác giả. |
| 8.5 | tests/test_cars.py | So trie mass và phân phối accepted với đáp án liệt kê chính xác; kiểm tra parser pruning và full-leaf rejection không loại đầu ra reward 1. Kiểm tra xác suất không bị cắt theo K của shortlist MDM-VGB. |
| 8.6 | src/tasks/generate_cars.py — T09 | Load MDLM hoặc AR theo suite, chạy sampler, ghi mọi attempt, thu mẫu unique bằng molecular identity. Prompt duplicates/uniqueness là quy tắc chọn mẫu bên ngoài reward, không tự biến thành failed leaves của tau. |
| 8.7 | modal/generate_cars.py — T09 | Chạy cùng sampler và recorder trên Modal; hỗ trợ config chính hoặc replication. |

## 9) Value providers, MDM-VGB và hybrid

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 9.1 | src/values.py | Định nghĩa một ValueProvider interface và hoàn thiện cả learned provider lẫn CARS provider. Lá hoàn chỉnh trả tau thật; state chưa hoàn chỉnh mới gọi estimator. Cache có giới hạn và khóa đủ context; quy tắc floor cho guidance không thay reward thật ở lá. |
| 9.2 | src/reference.py | Monte Carlo độc lập theo q_z và rollout tham chiếu từ cùng masked state, cùng nhóm/known tokens. Ghi policy, số completion và khoảng bất định; không gọi giá trị từ rollout policy là conditional lý tưởng nếu chưa chứng minh. |
| 9.3 | tests/test_values.py | Kiểm tra p_t*tau(Y_t), mọi attempt kể cả thất bại, freeze trie trước draw/update sau attempt, và giá trị exact trên phân phối hữu hạn. Kiểm tra terminal anchoring và phân biệt V_q với value theo policy khác. |
| 9.4 | configs/mdm_vgb.yaml | Khởi đầu L_f=8, L_b=32, K=8, B=1, lambda=0, temperature 1, forward random, backward low_conf, N=16; grid N=1/2/4/8/16. Chi=1 chỉ áp dụng khi bật Momentum; primary dùng kernel thường. |
| 9.5 | src/mdm_vgb.py | Triển khai forward/backward proposals, shortlist, trọng số chuyển trạng thái, remasking và terminal handling. Nhận ValueProvider qua hợp đồng đã có; không hard-code learned verifier. Budget nN là số bước tìm kiếm, không phải NFE; fallback forward được ghi chi phí riêng. |
| 9.6 | configs/hybrid.yaml | Dùng cùng kernel, MDLM, tokenizer, reward và setting search như MDM-VGB. Khác provider; m=16, khảo sát m=4/8/16/32, cache capacity và compute budget. Giữ m cố định ở primary estimator. |
| 9.7 | src/hybrid.py | Ghép CARSValueProvider vào kernel MDM-VGB đã hoàn thiện. Không viết kernel thứ hai. Thống nhất budget giữa outer search và inner completion; khi budget hết phải ghi trạng thái kết thúc, không gọi thêm value miễn phí. |
| 9.8 | configs/base.yaml | Chốt baseline unguided, class conditioning, completion policy, length và stopping rules; dùng cùng checkpoint với ba phương pháp chính. |
| 9.9 | src/tasks/generate_base.py — T08 | Sinh baseline bằng sampler MDLM tham chiếu, dùng cùng reward, selector và định dạng logs. Giải quyết đo độ khó ban đầu của từng nhóm. |
| 9.10 | modal/generate_base.py — T08 | Wrapper baseline trên Modal. |
| 9.11 | src/tasks/generate_mdm_vgb.py — T10 | Load MDLM + learned verifier, tạo provider và chạy kernel. Kiểm tra compatibility giữa verifier, tokenizer, reward và model. |
| 9.12 | modal/generate_mdm_vgb.py — T10 | Wrapper MDM-VGB trên Modal. |
| 9.13 | src/tasks/generate_hybrid.py — T11 | Load cùng MDLM, tạo CARS provider và chạy hybrid. Không yêu cầu learned-verifier checkpoint cho chính nguồn value của hybrid. |
| 9.14 | modal/generate_hybrid.py — T11 | Wrapper hybrid trên Modal; tính mọi inner forward vào cùng recorder. |
| 9.15 | tests/test_integration.py | Trên cùng state/proposals và một provider có cùng giá trị, baseline/hybrid phải dùng cùng kernel và cho hành vi tương ứng. Kiểm tra known tokens, EOS/PAD, terminal acceptance, budget exhaustion, resume và không double-count NFE. |

### Công thức cần hoàn thiện trong src/values.py

Với W_t chỉ chứa nhánh chắc chắn có tau=0, p_t=q_z(tránh W_t). Mỗi attempt lấy Y_t từ q_z có điều kiện tránh W_t đã chốt trước attempt. Dùng:

$$
\widehat V_q(z,c)=\frac{1}{m}\sum_{t=1}^{m}p_t\,\tau_c(Y_t).
$$

Mẫu chưa được kiểm tra cũng như attempt thất bại đều thuộc quá trình ước lượng. Khi một prefix đã chứng minh thất bại, có thể dừng attempt đó và đóng góp 0. Không dùng tỷ lệ thành công trên danh sách accepted-only, không bỏ hệ số mass và không thay full vocabulary bằng K=8. Dừng sớm dựa trên kết quả quan sát hoặc thay đổi policy là biến thể riêng; tính không chệch của trung bình với m cố định không tự động áp dụng.

## 10) Đánh giá trực tiếp việc xấp xỉ value

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 10.1 | configs/value_eval.yaml | Chốt states test độc lập, nhóm, tỷ lệ mask, seed và ngân sách Monte Carlo/estimator. So sánh learned provider với CARS provider tại cùng compute và cùng mục tiêu q_z; rollout tham chiếu được đo riêng. |
| 10.2 | src/tasks/evaluate_value.py — T13 | Tạo/lưu state test, tính MC reference, prediction, MAE/MSE, calibration và chi phí. Nếu reference chưa đủ độ chính xác, báo uncertainty; không coi MC estimate hữu hạn là giá trị exact. |
| 10.3 | modal/evaluate_value.py — T13 | Chạy batch value queries/MC trên GPU, lưu shard kết quả và ghép có kiểm soát. Không trộn states dùng chọn hyperparameter với states đánh giá cuối. |

## 11) Metrics, retrosynthesis và báo cáo

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 11.1 | src/metrics.py | Tính success trên root attempts, validity/membership trên đầu ra hoàn chỉnh, unique/novel molecular identity, diversity trên valid molecules và chi phí. Báo mẫu số, số mẫu thực đạt và trường hợp thiếu dữ liệu; không báo validity 100% trên selected successes như bằng chứng hiệu quả. |
| 11.2 | src/retrosynthesis.py | Load reaction model theo manifest T04, tính đúng proxy gốc: valid molecule có phản ứng top-1 với probability >0,4. Công bố denominator; không gọi proxy này là chứng minh có lộ trình tổng hợp nhiều bước. Model thiếu thì báo metric unavailable, không thay bằng 0. |
| 11.3 | configs/evaluation.yaml | Khóa reward, tập mẫu áp dụng, Morgan radius 2/2.048 bit và công thức RetroStar proxy. Radius 3 là sensitivity theo mã gốc. Loại QED/Lipinski khỏi chỉ số chính; lưu cả attempts, full candidates và selected set. |
| 11.4 | tests/test_metrics.py | Kiểm tra mẫu số với prefix abort/duplicate/invalid, Tanimoto trên bộ nhỏ có đáp án biết trước, unique theo molecule và chi phí inner completion. Dùng stub reaction outputs để kiểm tra công thức retro mà không download model chỉ để unit test. |
| 11.5 | src/tasks/evaluate.py — T14 | Đọc logs đã hoàn thành, xác nhận protocol/model/reward metadata và ghi metrics.json. Không thay đổi samples/attempts đầu vào; gặp run thiếu phải ghi trạng thái thay vì âm thầm bỏ. |
| 11.6 | modal/evaluate.py — T14 | Wrapper đánh giá với tài nguyên phù hợp cho RDKit/retrosynthesis, lưu metric vào run tương ứng. |
| 11.7 | src/reporting.py | Tổng hợp per-group/per-seed, paired comparisons, khoảng tin cậy và quality–cost frontiers. Tách primary, replication và smoke; kết quả timeout vẫn xuất hiện với số mẫu thực tế. |
| 11.8 | configs/report.yaml | Chốt bảng, biểu đồ, nhóm metric và output filenames. Hiển thị riêng độ chính xác value, chất lượng phân tử và chi phí; không gộp hai model khác nhau thành một so sánh cùng backbone. |
| 11.9 | src/tasks/report.py — T16 | Xuất CSV, PNG và report.md từ metrics đã được kiểm tra. Báo missing/failed runs, checkpoint/config/source và giới hạn V_q so với V*. |
| 11.10 | modal/report.py — T16 | Wrapper tổng hợp trên Volume và trả đường dẫn báo cáo; có thể tải artifact về local sau khi hoàn tất. |

## 12) Điều phối so sánh và hướng dẫn sử dụng

| Thứ tự | Tệp | Chức năng, vấn đề giải quyết và tiêu chí hoàn thành |
|---|---|---|
| 12.1 | configs/comparison.yaml | Định nghĩa matrix ba phương pháp chính × ba nhóm × ít nhất ba seed × compute budgets; baseline là đối chiếu bổ sung. So sánh theo ngân sách NFE/thời gian tương ứng, tính cả inner completion của hybrid. Giữ run timeout hoặc chưa đạt mục tiêu với số mẫu thực tế; không ngoại suy hay loại bỏ. Suite replication riêng. Tách validation sweep với test cấu hình đã khóa. |
| 12.2 | src/tasks/compare.py — T15 | Kiểm tra dependencies trước chạy, tạo job manifest, gọi các task dùng chung trên local và resume job chưa hoàn thành. Xác nhận checkpoint/tokenizer/reward chung trong primary suite; lưu cả run không đạt 100 mẫu. Không tự huấn luyện lại hoặc đổi checkpoint giữa các phương pháp. |
| 12.3 | modal/compare.py — T15 | Điều phối job theo method/group/seed/budget và dispatch worker với tài nguyên riêng. Dùng cùng task functions; không chạy toàn bộ coordinator như một job GPU duy nhất. Mỗi worker ghi thư mục run riêng, commit xong mới đánh giá/tổng hợp. |
| 12.4 | README.md | Hoàn thiện hướng dẫn môi trường, T01–T16, config overrides, local/Modal, Secret cho model cần quyền, artifact/resume và paper differences. Lệnh, đường dẫn và thuật ngữ phải khớp các tệp đã hoàn thiện. |

## 13) Các tệp artifact tự sinh

| Đường dẫn / tên tệp | Task tạo | Nội dung và mục đích |
|---|---|---|
| requirements.lock | Tùy chọn, tự sinh từ pyproject.toml | Ghi phiên bản/checksum dependencies cho Linux/Python 3.12 nếu cần; image Modal cài dependencies trực tiếp từ pyproject.toml |
| data/benchmark/examples.parquet | T01 | Ví dụ gốc, nhóm, canonical identity và kết quả từng checker |
| data/benchmark/manifest.json | T01 | Provenance/checksum/counts và mismatch đã ghi nhận |
| data/corpus/train.parquet | T02 | Corpus huấn luyện; molecule identity và multi-label |
| data/corpus/validation.parquet | T02 | Dữ liệu chọn cấu hình/checkpoint |
| data/corpus/test.parquet | T02 | Dữ liệu đã khóa để kiểm tra cuối và xây states test |
| data/corpus/manifest.json | T02 | Nguồn/version, filters, seed, split và thống kê |
| data/tokenizer/vocab.json | T03 | Token vocabulary và các special token IDs |
| data/tokenizer/metadata.json | T03 | Quy tắc tokenize, max length, hash và coverage |
| data/tokenized/<split>/shard_<id>.pt | T03 | Tokenized examples/labels theo shard |
| data/tokenized/<split>/manifest.json | T03 | Shard list, source/tokenizer hashes và counts |
| data/rollouts/<run_id>/shard_<id>.pt | T06 | State snapshots, group, terminal identity/reward và partition |
| data/rollouts/<run_id>/manifest.json | T06 | Model/reward/policy hashes và cấu hình rollout |
| data/value_eval/<run_id>/states.pt | T13 | Bộ masked states độc lập có ID và nguồn |
| data/value_eval/<run_id>/reference_values.parquet | T13 | MC reference theo từng policy, uncertainty, provider estimates và chi phí |
| checkpoints/pretrained/<model_id>/<revision>/ | T04 | Tệp snapshot của nhà cung cấp; danh sách thực tế theo snapshot manifest, không viết tay hay import từ resource |
| checkpoints/<mdlm_or_verifier>/<run_id>/<step>/model.safetensors | T05/T07 | Trọng số model của checkpoint cụ thể |
| checkpoints/<mdlm_or_verifier>/<run_id>/<step>/training_state.pt | T05/T07 | Optimizer/scheduler/RNG/step cho resume |
| checkpoints/<mdlm_or_verifier>/<run_id>/<step>/metadata.json | T05/T07 | Model config, tokenizer/corpus/rollout hashes, validation và smoke flag |
| outputs/.../resolved_config.yaml | Mọi task tạo run | Cấu hình thực tế sau defaults/overrides/path resolution |
| outputs/.../manifest.json | Mọi task tạo run | Task ID, run status, artifact references, hashes và lỗi nếu có |
| outputs/.../attempts.jsonl | T08–T11 | Root attempts, complete output hoặc prefix abort, lý do kết thúc và counters |
| outputs/.../samples.jsonl | T08–T11 | Các đầu ra được selector nhận, canonical identities và provenance |
| outputs/.../events.jsonl | Task được tracking | Leaf visits, transitions, value queries, cache hits, compute và diagnostics kiểm tra |
| outputs/.../metrics.json | T12–T14 | Kết quả kiểm tra/đánh giá, số mẫu, denominator và trạng thái unavailable |
| reports/<suite_id>/summary.csv | T16 | Metrics theo method/group/seed/budget và tổng hợp |
| reports/<suite_id>/value_accuracy.csv | T16 | Sai số/calibration/chi phí theo provider, nhóm, mask depth và policy |
| reports/<suite_id>/quality_cost.png | T16 | Chất lượng theo raw/adjusted NFE và thời gian |
| reports/<suite_id>/value_calibration.png | T16 | Calibration plot kèm số state và uncertainty tham chiếu |
| reports/<suite_id>/report.md | T16 | Thiết lập, kết quả, paper differences, missing runs và giới hạn kết luận |

Trên local, artifact root mặc định là project root. Trên Modal, dùng Volume smiles-artifacts mount ở /workspace; cùng cấu trúc data/checkpoints/outputs/reports. Dữ liệu/model không được trả nguyên khối qua RPC; task trả status/manifest/path. Mỗi worker ghi run/shard riêng, commit sau ghi và reload trước đọc artifact mới từ worker khác.

Nguồn đối chiếu: PLAN.md; resource/2510.01902v2.pdf, đặc biệt Phụ lục C/D/H; resource/2606.28301v1.pdf, mục 5 và Phụ lục F; các repository được cung cấp dùng để tham khảo; [benchmark CARS của tác giả](https://github.com/pparys/cars/tree/f46607f983d89e4ca4f40527cd231ab04adc6ed7/datasets/smiles). Thiết kế wrapper/CLI dựa trên [Modal Apps và entrypoints](https://modal.com/docs/guide/apps); artifact persistence dựa trên [Modal Volumes](https://modal.com/docs/guide/volumes).
