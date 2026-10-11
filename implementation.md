# Thứ tự triển khai CARS và MDM-VGB trên QM9

Tài liệu STEP 5 quy định thứ tự implement từng file, nhiệm vụ của file, vấn đề cần giải quyết và điều kiện hoàn thành. Đây vẫn là kế hoạch: bước viết tài liệu này không tạo source/script, tải assets, train model hoặc chạy job Modal.

Tên tài liệu chính thức là implementation.md; phần cây thư mục được đặt ngay trong tài liệu này, không tạo thêm một bản implement.md trùng nội dung.

Đọc cùng [PLAN.md](PLAN.md) về phương pháp và [modal/README.md](modal/README.md) về các task Modal. Code tham khảo trong resource chỉ được đọc. Phần cần dùng lại phải được sao chép hợp lệ vào package độc lập, ghi nguồn/license và đổi import; không sửa/import/cài/mount/upload resource trong runtime.

**Quy tắc thực hiện**

- Đi theo thứ tự 1.1 → 1.2 → … → 14.5 trong các bảng. Hoàn thiện chức năng của file trước khi sang file kế tiếp; không dùng pass, NotImplementedError hoặc nhánh giả để gọi là đã xong.
- File chỉ import các module dự án đã hoàn thành trước nó. Các protocol được định nghĩa sớm; factory và entrypoint tổng hợp đặt cuối để tránh phụ thuộc vòng. __init__.py tối thiểu là một file hoàn chỉnh theo đúng nhiệm vụ, không cần re-export mọi module.
- Kiểm tra ngay điều kiện hoàn thành của từng file và cổng kiểm chứng cuối giai đoạn. File test đứng sau module mà nó kiểm tra; chạy kiểm chứng nhỏ trước khi triển khai tiếp phần phụ thuộc.
- Nếu kiểm chứng phát hiện lỗi, sửa file gây lỗi trước khi tiếp tục. Thứ tự tuyến tính không có nghĩa cấm sửa lỗi hoặc cập nhật tài liệu theo bằng chứng.
- Mỗi task Modal có một file modal/<task>.py và một script modal/run_<task>.sh gọi modal run modal/<task>.py::main. Không tạo modal/__init__.py; thuật toán nằm trong src/cars_vgb.
- Phân biệt thứ tự viết file với thứ tự chạy task. CARS-rho đánh giá trạng thái online bằng discovery → frozen-trie estimation → cache; không cần train neural verifier. Neural training là nhánh baseline riêng.
- Target ban đầu là V_rho theo completion q_rho. Chỉ gọi là xấp xỉ V* khi reference joint/conditional và điều kiện consistency đã xác định; giữ V_rho và V* của reference kiểm chứng thành hai nhãn.
- Quy mô pilot/benchmark phải có nguồn hoặc được chốt riêng trước đánh giá cuối. Các số 128 và 3 seed × 1.000 output trong đề xuất ban đầu của PLAN chưa phải protocol paper đã được xác minh.

**Cây cấu trúc sẽ triển khai**

Các file source/config/test/docs dưới đây có mục riêng trong thứ tự triển khai. Nội dung bên trong resource giữ nguyên và được thu gọn ở cấp repository tham khảo. data, model_data, cache và results là artifact được sinh khi chạy, không phải file source cần viết trước.

```text
sampling/
├── README.md
├── PLAN.md
├── implementation.md
├── pyproject.toml
├── .gitignore
├── THIRD_PARTY_NOTICES.md
├── resource/                           # Chỉ đọc tham khảo; không import/mount/upload
│   ├── Constrained Adaptive Rejection Sampling.pdf
│   ├── VGB for Masked Diffusion Model Efficient Test-time Scaling for Reward Satisfaction and Sample Editing.pdf
│   ├── cars-main/
│   └── MDM-VGB-main/
├── modal/
│   ├── README.md
│   ├── prepare_assets.py
│   ├── run_prepare_assets.sh
│   ├── train_base.py
│   ├── run_train_base.sh
│   ├── calibrate.py
│   ├── run_calibrate.sh
│   ├── collect_rollouts.py
│   ├── run_collect_rollouts.sh
│   ├── train_verifier.py
│   ├── run_train_verifier.sh
│   ├── prepare_value_panel.py
│   ├── run_prepare_value_panel.sh
│   ├── generate.py
│   ├── run_generate.sh
│   ├── check_values.py
│   ├── run_check_values.sh
│   ├── benchmark.py
│   ├── run_benchmark.sh
│   ├── evaluate.py
│   ├── run_evaluate.sh
│   ├── verify.py
│   └── run_verify.sh
├── configs/
│   ├── qm9.yaml
│   ├── base_training.yaml
│   ├── rollouts.yaml
│   ├── verifier_training.yaml
│   ├── mdm_vgb.yaml
│   ├── cars.yaml
│   ├── benchmark.yaml
│   ├── reference_checks.yaml
│   └── modal.yaml
├── src/
│   └── cars_vgb/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── types.py
│       ├── randomness.py
│       ├── io.py
│       ├── budget.py
│       ├── execution/
│       │   ├── __init__.py
│       │   └── modal_runtime.py
│       ├── qm9/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── chemistry.py
│       │   ├── reward.py
│       │   ├── calibration.py
│       │   └── rollouts.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── configuration_udlm.py
│       │   ├── modeling_udlm.py
│       │   └── backbone.py
│       ├── sampling/
│       │   ├── __init__.py
│       │   ├── native.py
│       │   ├── completion.py
│       │   ├── reference.py
│       │   ├── base.py
│       │   ├── bon.py
│       │   ├── rejection.py
│       │   ├── candidates.py
│       │   ├── vgr.py
│       │   ├── vgb.py
│       │   ├── momentum.py
│       │   └── terminal.py
│       ├── cars/
│       │   ├── __init__.py
│       │   ├── trie.py
│       │   ├── oracle.py
│       │   ├── proposal.py
│       │   ├── estimator.py
│       │   └── cache.py
│       ├── verifiers/
│       │   ├── __init__.py
│       │   ├── interface.py
│       │   ├── neural.py
│       │   ├── monte_carlo.py
│       │   ├── cars.py
│       │   └── importance.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── verifier.py
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── run.py
│       │   ├── benchmark.py
│       │   └── value_checks.py
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py
│           ├── aggregate.py
│           └── plots.py
├── tests/
│   ├── conftest.py
│   ├── test_qm9_protocol.py
│   ├── test_token_rules.py
│   ├── test_trie.py
│   ├── test_oracle.py
│   ├── test_cars_proposal.py
│   ├── test_value_estimator.py
│   ├── test_importance.py
│   ├── test_vgb.py
│   ├── test_cache.py
│   ├── test_budget.py
│   ├── test_pipeline.py
│   ├── test_no_resource_dependency.py
│   └── fixtures/
│       ├── toy_joint.json
│       └── toy_grammar.lark
├── data/
│   ├── qm9/
│   │   ├── raw/
│   │   └── splits.json
│   ├── calibration/
│   │   └── <calibration_id>/
│   │       ├── samples.jsonl
│   │       └── calibration.json
│   ├── rollouts/
│   │   └── <rollout_id>/
│   │       ├── states.pt
│   │       └── manifest.json
│   └── value_panels/
│       └── <panel_id>/
│           ├── states.jsonl
│           └── manifest.json
├── model_data/
│   ├── base/
│   │   └── <checkpoint_id>/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── training_state.pt         # Optimizer/scheduler/RNG khi train/resume
│   │       ├── tokenizer/
│   │       └── manifest.json
│   └── verifier/
│       └── <checkpoint_id>/
│           ├── verifier.pt
│           ├── training_state.pt         # Optimizer/scheduler/RNG khi train/resume
│           └── manifest.json
├── cache/
│   └── tries/
│       └── <cache_id>/
│           ├── trie.npz
│           ├── values.jsonl
│           └── metadata.json
└── results/
    ├── raw/
    │   └── <run_id>/
    │       ├── config.yaml
    │       ├── manifest.json
    │       ├── samples.jsonl
    │       ├── values.jsonl
    │       ├── costs.jsonl
    │       └── tests.txt
    └── summary/
        └── <comparison_id>/
            ├── metrics.csv
            ├── metrics.json
            ├── report.md
            └── figures/
                ├── quality_cost.png
                ├── value_error_cost.png
                └── cache_cost.png
```

Trên Modal, các thư mục artifact được lưu dưới /workspace trên Volume dùng chung. Các placeholder <...> là ID thực khi chạy; training_state.pt chỉ có khi tạo checkpoint hỗ trợ train/resume. Phần mô tả artifact ở cuối tài liệu giải thích module nào sinh từng nhóm file.

**Thứ tự implement từng file**

**1) Chuẩn bị môi trường, package và các hợp đồng dùng chung**

Chốt kiểu dữ liệu và ranh giới module trước khi viết thuật toán. Các __init__.py chỉ khai báo package, không import module chưa tồn tại và không tạo tài nguyên cloud khi import.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 1.1 | `pyproject.toml` | Khai báo package src/cars_vgb, phiên bản Python và dependencies tương thích cho Torch, Transformers, RDKit, Modal, dữ liệu, kiểm chứng và biểu đồ. Giải quyết khác biệt môi trường local/Modal. | Cài package và import dependencies trong môi trường sạch; không cài repository từ resource. |
| 1.2 | `.gitignore` | Bỏ qua checkpoint, dataset, cache, artifact lớn và thông tin xác thực. Giữ source, cấu hình và tài liệu trong Git. | Các đường dẫn artifact và credential được loại đúng, còn source/config/tests vẫn được theo dõi. |
| 1.3 | `THIRD_PARTY_NOTICES.md` | Ghi nguồn, revision/hash, license và các thay đổi của phần code sẽ sao chép. Giải quyết provenance khi tách khỏi repository tham khảo. | Có danh sách nguồn dự kiến; chỉ sao chép phần được phép, viết mới phần chưa xác minh quyền sử dụng. |
| 1.4 | `src/cars_vgb/__init__.py` | Khai báo package gốc và metadata tối thiểu; tránh import tạo model hoặc kết nối Modal. | Import cars_vgb không cần dataset, checkpoint hoặc tài khoản Modal. |
| 1.5 | `src/cars_vgb/execution/__init__.py` | Khai báo package điều phối thực thi. | Import package không tạo image, Volume hoặc job. |
| 1.6 | `src/cars_vgb/qm9/__init__.py` | Khai báo package dữ liệu và reward QM9. | Không tải dataset khi import. |
| 1.7 | `src/cars_vgb/models/__init__.py` | Khai báo package kiến trúc và adapter backbone. | Không load trọng số khi import. |
| 1.8 | `src/cars_vgb/sampling/__init__.py` | Khai báo package sampler, không dựng registry bằng các import chưa tồn tại. | Import độc lập thành công. |
| 1.9 | `src/cars_vgb/cars/__init__.py` | Khai báo package trie và estimator CARS. | Không khởi tạo trie toàn cục khi import. |
| 1.10 | `src/cars_vgb/verifiers/__init__.py` | Khai báo package các nguồn giá trị trạng thái. | Không yêu cầu neural checkpoint khi import. |
| 1.11 | `src/cars_vgb/training/__init__.py` | Khai báo package training. | Import không bắt đầu training. |
| 1.12 | `src/cars_vgb/experiments/__init__.py` | Khai báo package điều phối thí nghiệm. | Import không bắt đầu run hoặc benchmark. |
| 1.13 | `src/cars_vgb/evaluation/__init__.py` | Khai báo package đánh giá artifact. | Import không tải model hoặc sửa kết quả. |
| 1.14 | `src/cars_vgb/types.py` | Định nghĩa MaskedState, SampleResult, ValueEstimate, cost records và protocol cho backbone/sampler. Giải quyết việc các module diễn giải trạng thái hoặc target khác nhau. | Hợp đồng có token IDs, mask/token rules, target/reference ID, trạng thái lỗi và chi phí; hỗ trợ backbone giả mà không cần model thật. |
| 1.15 | `src/cars_vgb/config.py` | Đọc YAML, resolve đường dẫn/override, kiểm tra schema và lưu nguồn từng setting: paper, code upstream hoặc đề xuất dự án. Giải quyết cấu hình thiếu và target mismatch. | Phát hiện cấu hình mâu thuẫn; không tự suy ngưỡng reward hoặc tự chọn số seed/output. |
| 1.16 | `src/cars_vgb/randomness.py` | Tách RNG cho sampler, discovery, estimation và từng run. Giải quyết việc batching hoặc query lồng làm thay đổi ngẫu nhiên ngoài ý muốn. | Có seed dẫn xuất ổn định và cơ chế lưu/khôi phục RNG. |
| 1.17 | `src/cars_vgb/io.py` | Đọc/ghi JSONL, YAML, checkpoint metadata, hash và manifest theo run/attempt ID. Giải quyết ghi đè và artifact chưa hoàn tất. | Ghi hoàn tất có dấu xác nhận; lỗi ghi không biến artifact dở thành artifact hợp lệ. |
| 1.18 | `src/cars_vgb/budget.py` | Đếm forward batch thật, sequence work, lượt verifier và chi phí theo giai đoạn; quản lý phần ngân sách dự phòng completion. Giải quyết bỏ sót rollout lồng. | API nhận event từ model thật, không tăng NFE chỉ vì backtrack; hỗ trợ reserve toàn query trước khi chạy. |

Cổng kiểm chứng: Có package độc lập và hợp đồng đủ để các file sau chỉ phụ thuộc phần đã hoàn thành. Chưa cần checkpoint hoặc chạy GPU.

**2) Dữ liệu, reward và task Modal đầu tiên**

Hoàn thiện phần dữ liệu trước khi train model. Task prepare_assets chuẩn bị dataset/tokenizer, không tự tạo ra trọng số pretrained.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 2.1 | `configs/qm9.yaml` | Chốt dataset/tokenizer ID và revision, split, chiều dài 32, token đặc biệt và quy tắc canonicalization. Giải quyết khác biệt đầu vào giữa phương pháp. | Mỗi giá trị có nguồn; vocabulary phải lấy từ assets thực, không đoán từ bảng hoặc default verifier. |
| 2.2 | `configs/modal.yaml` | Khai báo CPU/GPU, memory, timeout, concurrency, image và Volume/path dùng chung. Giải quyết cấu hình hạ tầng bị lặp giữa các task. | Tách tài nguyên khỏi siêu tham số thuật toán; cấu hình được kiểm tra trước khi tạo job. |
| 2.3 | `src/cars_vgb/qm9/chemistry.py` | Parse SMILES, canonicalize, tính QED và molecular identity bằng RDKit. Giải quyết novelty/dedup sai vì cùng phân tử có nhiều SMILES. | Invalid có kết quả rõ ràng; train và output dùng cùng cách chuẩn hóa và phiên bản RDKit. |
| 2.4 | `src/cars_vgb/qm9/dataset.py` | Tải/cache QM9 và tokenizer; lưu ID split 5% held-out với split seed 42 theo code tham khảo, hash và tập train canonical. Giải quyết split thay đổi giữa run. | Có manifest revision, split IDs và đối chiếu kích thước thực; không dùng held-out làm prompt generation vô điều kiện. |
| 2.5 | `src/cars_vgb/qm9/reward.py` | Tách điểm liên tục R=QED hoặc 0 khỏi reward nhị phân Pass@95. Kiểm tra hoàn chỉnh trước decode, validity và threshold. | MASK còn lại không thể thành công do bị xóa khi decode; threshold chưa có được báo là chưa calibration, không giả thành reward 0. |
| 2.6 | `src/cars_vgb/execution/modal_runtime.py` | Dùng cấu hình hạ tầng để định nghĩa image, dependencies, Volume mount /workspace và các thao tác persist/read artifact. Giải quyết môi trường và storage dùng chung. | Chỉ đóng gói source/config/metadata/tests cần thiết; không upload, mount hoặc cài resource; commit/reload và đường dẫn attempt rõ ràng. |
| 2.7 | `modal/prepare_assets.py` | Entry point main nhận config, gọi dataset preparation trên CPU và trả manifest/đường dẫn artifact. | Chạy task tạo dữ liệu/tokenizer trên Volume, lỗi tải hoặc revision phải hiển thị; không báo đã có backbone khi chưa có weights. |
| 2.8 | `modal/run_prepare_assets.sh` | Gọi modal run modal/prepare_assets.py::main từ project root và chuyển tiếp tham số. | Hoạt động từ thư mục gọi khác, giữ nguyên tham số và exit code; không chứa credential. |

Cổng kiểm chứng: Sau khi bắt đầu IMPLEMENT mới thiết lập tài khoản Modal và chạy task prepare_assets. Kiểm tra dataset, tokenizer và manifest tồn tại trên Volume trước training.

**3) Kiến trúc độc lập và huấn luyện backbone**

Dùng kiến trúc QM9 MDLM làm backbone chung. Sao chép phần kiến trúc cần thiết vào package dự án với license/provenance, không load code qua resource.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 3.1 | `src/cars_vgb/models/configuration_udlm.py` | Định nghĩa UDLMConfig trong package nội bộ; giữ các trường kiến trúc tương thích checkpoint. Giải quyết phụ thuộc auto_map vào source bên ngoài. | Config lưu/nạp lại được và không trỏ tới resource. |
| 3.2 | `src/cars_vgb/models/modeling_udlm.py` | Định nghĩa kiến trúc UDLM phục vụ objective MDLM, sửa relative imports sang package nội bộ. Giải quyết khả năng training/inference độc lập. | Forward/backward trên model nhỏ chạy được; shape, timestep conditioning và tên trọng số tương thích thiết kế đã chọn. |
| 3.3 | `src/cars_vgb/models/backbone.py` | Load hoặc khởi tạo model, load tokenizer, kiểm tra vocabulary và expose logits/token log-probabilities theo contract; gắn instrumentation tại forward thật. | Checkpoint round-trip và batching đúng; vocabulary/token mismatch bị từ chối; chi phí của mọi caller được ghi nhận. |
| 3.4 | `tests/conftest.py` | Cung cấp fixture tokenizer/model nhỏ, backbone giả và thư mục artifact tạm. Giải quyết việc kiểm chứng phải tải model 92,4M tham số. | Fixtures độc lập mạng, nhận phân phối toy từ test và không import module chưa implement. |
| 3.5 | `tests/test_token_rules.py` | Kiểm tra MASK/EOS/PAD/UNK, trạng thái hoàn chỉnh, token đã reveal và shape/vocabulary. Giải quyết lỗi chuỗi bị decode thành thành công giả. | Bao phủ mask có lỗ và các trường hợp decode che lỗi completion. |
| 3.6 | `configs/base_training.yaml` | Ghi kiến trúc, objective MDLM, optimizer, schedule, precision, batch và checkpoint policy. Giải quyết training khác nguồn mà không được khai báo. | Các default như seed 1 và 25.000 bước được gắn nhãn code upstream; không tuyên bố là toàn bộ setting paper đã xác minh. |
| 3.7 | `src/cars_vgb/training/base.py` | Train/resume MDLM từ split train; lưu weights, optimizer, scheduler, bước training và RNG. Giải quyết thiếu checkpoint inference có provenance. | Một lượt train nhỏ và resume giữ đầy đủ trạng thái; dataset/tokenizer/config hashes đi kèm checkpoint. |
| 3.8 | `modal/train_base.py` | Entry point GPU cho training/resume backbone, chọn checkpoint/output ID và persist checkpoint trên Volume. | Checkpoint đọc được bằng backbone adapter ở task khác; task thất bại không để lại manifest hoàn thành giả. |
| 3.9 | `modal/run_train_base.sh` | Gọi modal run modal/train_base.py::main với config và tham số resume. | Giữ exit code, không tự training lại khi người dùng chỉ yêu cầu resume. |

Cổng kiểm chứng: Vượt kiểm chứng nhỏ trước training QM9 thật. Nếu có checkpoint tương thích với provenance xác minh được, dùng checkpoint đó sau kiểm tra loader; mọi phương pháp dùng cùng hash backbone và tokenizer.

**4) Sampler nền, reference và calibration**

Tách native MDLM khỏi completion q_rho. Hoàn thiện raw generation trước khi xác định threshold; calibration không phụ thuộc một reward nhị phân chưa được khóa.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 4.1 | `tests/fixtures/toy_joint.json` | Lưu joint nhỏ, reward và các masked state có thể liệt kê đầy đủ. Giải quyết việc không có ground truth chính xác cho kiểm chứng xác suất. | Tổng xác suất bằng 1; có prefix, mask có lỗ, trạng thái terminal và các completion reward 0/1. |
| 4.2 | `configs/reference_checks.yaml` | Chỉ định V_rho, joint pi_rho, reveal order và cấu hình enumeration/MC/importance. Giải quyết gọi nhầm rollout value là V*. | Mỗi target có reference ID và support assumptions; toy và QM9 được tách rõ. |
| 4.3 | `src/cars_vgb/sampling/native.py` | Triển khai schedule và posterior của native MDLM, giữ token đã biết và thu snapshot khi được yêu cầu. Giải quyết tái tạo baseline diffusion. | Đường suy luận khớp công thức tham khảo; số bước và quy tắc completion được ghi rõ, forward cache không làm sai accounting. |
| 4.4 | `src/cars_vgb/sampling/completion.py` | Hoàn tất trạng thái theo thứ tự rho cố định, giữ nguyên mọi token đã reveal. Giải quyết completion có phân phối q_rho xác định để so sánh MC/CARS. | Sinh token và tính log q_rho nhất quán; arbitrary masks giữ đúng ngữ cảnh tương lai đã biết. |
| 4.5 | `src/cars_vgb/sampling/reference.py` | Định nghĩa joint pi_rho từ root và tính full-sequence likelihood theo prefix gốc. Giải quyết likelihood bị nhầm với q_rho có token tương lai cố định. | Likelihood khớp tổng/enumeration trên toy; không tuyên bố pi_rho là native diffusion joint. |
| 4.6 | `src/cars_vgb/sampling/base.py` | Cung cấp Base-native và Base-rho qua cùng contract kết quả và chi phí. Giải quyết baseline thay đổi cả reference mà không ghi nhận. | Output và manifest phân biệt hai sampler; raw generation chạy được trước khi có threshold. |
| 4.7 | `src/cars_vgb/qm9/calibration.py` | Sinh calibration độc lập từ base đã đóng băng, tính percentile 95 của R và lưu threshold/provenance. Giải quyết dùng nhầm percentile dataset hoặc dùng tập đánh giá. | Invalid có R=0; lưu sampler, seed, số mẫu, quy tắc quantile và bất định; khóa threshold ID cho các run sau. |
| 4.8 | `tests/test_qm9_protocol.py` | Kiểm tra split cố định, canonical identity, reward và tính độc lập của calibration bằng dữ liệu nhỏ. Giải quyết leakage và mẫu số không thống nhất. | Không cần tải toàn QM9; kiểm tra trực tiếp ID, quantile trên R gồm invalid bằng 0, threshold requirement và invalid/unfinished handling. |
| 4.9 | `modal/calibrate.py` | Entry point GPU gọi calibration và ghi calibration artifact vào Volume. | Threshold gắn đúng model/tokenizer/sampler; không ghi đè calibration đã khóa. |
| 4.10 | `modal/run_calibrate.sh` | Gọi modal run modal/calibrate.py::main và chuyển tiếp tham số calibration. | Có thể chỉ định checkpoint/calibration ID, giữ exit code. |

Cổng kiểm chứng: Có Base-native, Base-rho và threshold độc lập. Không mặc định tái sử dụng threshold số trong config upstream khi chưa xác minh nguồn calibration.

**5) Verifier interface, Monte Carlo và baseline neural**

Hoàn thiện các nguồn giá trị đối chứng trước khi thêm CARS. Neural verifier là baseline cần training riêng; MC và CARS không phải đi qua training neural.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 5.1 | `src/cars_vgb/verifiers/interface.py` | Chuẩn hóa values_for_states theo batch, target ID, terminal anchoring, uncertainty và cost. Giải quyết các verifier trả giá trị không cùng ý nghĩa. | Giữ đúng thứ tự state đầu vào; terminal trả reward chính xác mà không gọi model verifier. |
| 5.2 | `src/cars_vgb/verifiers/monte_carlo.py` | Ước lượng V_rho bằng số completion cố định từ q_rho, kèm interval và tổng chi phí. Giải quyết thiếu baseline trực tiếp cho estimator CARS. | Không dedup draw; không đổi số mẫu theo kết quả; thiếu ngân sách trả trạng thái unavailable để caller fallback. |
| 5.3 | `configs/rollouts.yaml` | Chỉ định native hoặc q_rho, số bước, snapshot và nhóm train/validation. Giải quyết target mismatch giữa label và inference. | Khác biệt rollout 128 bước và inference 32 bước, nếu giữ, được khai báo; số lượng dữ liệu không giả là setting paper. |
| 5.4 | `src/cars_vgb/qm9/rollouts.py` | Sinh trajectory, gắn terminal reward cho snapshot, lưu rollout ID và nhóm split. Giải quyết leakage giữa snapshot tương quan. | Tất cả snapshot cùng rollout thuộc một nhóm; label ghi target/policy/threshold ID. |
| 5.5 | `modal/collect_rollouts.py` | Entry point GPU tạo rollout shards và manifest cho verifier. | Shard có ID riêng và completion status; không trộn native và q_rho vào một target không khai báo. |
| 5.6 | `modal/run_collect_rollouts.sh` | Gọi modal run modal/collect_rollouts.py::main. | Chuyển tiếp config/seed/shard params và giữ exit code. |
| 5.7 | `src/cars_vgb/verifiers/neural.py` | Định nghĩa neural verifier và adapter load/batch inference. Giải quyết baseline learned process value và kiểm tra target checkpoint. | Kết quả đúng thứ tự, checkpoint metadata đúng target/vocabulary; không hardcode vocabulary 35 hoặc 40. |
| 5.8 | `configs/verifier_training.yaml` | Chốt kiến trúc, MSE objective, optimizer, split và checkpoint policy cho verifier. | Có cấu hình native-label và rho-label tách biệt; hyperparameter có nguồn. |
| 5.9 | `src/cars_vgb/training/verifier.py` | Train/validate neural verifier từ snapshot và terminal label; lưu training state để resume. Giải quyết baseline cùng target với CARS/MC. | Validation nhóm theo rollout; checkpoint có target, dữ liệu, threshold và RNG/optimizer state. |
| 5.10 | `modal/train_verifier.py` | Entry point GPU cho training/resume neural verifier. | Checkpoint có thể dùng ở task generation khác; không thay đổi backbone. |
| 5.11 | `modal/run_train_verifier.sh` | Gọi modal run modal/train_verifier.py::main. | Phân biệt training mới và resume; giữ config/target và exit code. |

Cổng kiểm chứng: Có neural-native, neural-rho và MC-rho với target rõ ràng. Kết luận regression xấp xỉ V* vẫn cần điều kiện reference consistency.

**6) Baseline BoN, VGR, VGB và Momentum**

Hoàn thiện graph và chính sách terminal trước khi thay bằng CARS verifier. Giữ kernel dùng cùng giao diện giá trị để tránh sửa thuật toán theo từng verifier.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 6.1 | `configs/mdm_vgb.yaml` | Ghi L_f, L_b, K, N, lambda, chi, shortlist và chính sách hết budget. Giải quyết default code bị hiểu nhầm là setting mọi thực nghiệm paper. | Có provenance; K là số candidate được lấy mẫu, không bị chuyển thành top-k truncation. |
| 6.2 | `src/cars_vgb/sampling/terminal.py` | Xử lý leaf success/failure, backtracking, zero-weight candidate và unguided fallback có reserve. Giải quyết thoát sớm vì validity hoặc hết budget. | Phân biệt valid với Pass@95; paper completion và upstream completion có nhãn riêng; unfinished/timeout được ghi thất bại. |
| 6.3 | `src/cars_vgb/sampling/candidates.py` | Xây candidate, multiplicity, depth scaling và trọng số forward/backward từ verifier interface. Giải quyết khác công thức giữa các phiên bản VGB. | Candidate hợp lệ giữ token rules; trọng số và lambda dùng đúng ngữ cảnh, không tự thêm epsilon tạo support mới. |
| 6.4 | `src/cars_vgb/sampling/bon.py` | Sinh N completion theo sampler đã chọn và chọn bằng terminal reward/QED theo protocol. | Tính đủ chi phí cả N completion; tie policy và unsuccessful output được ghi rõ. |
| 6.5 | `src/cars_vgb/sampling/vgr.py` | Triển khai value-guided resampling trên candidate dùng chung. Giải quyết baseline hướng dẫn tiến không backtrack. | Dùng verifier interface và accounting chung; terminal anchoring đúng. |
| 6.6 | `src/cars_vgb/sampling/vgb.py` | Triển khai edge walk với forward/backward, shortlist và budget/terminal policy. Giải quyết baseline MDM-VGB cần đối chiếu trước tích hợp. | Chạy được với neural và MC verifier; không tính backtrack thuần là model forward. |
| 6.7 | `src/cars_vgb/sampling/momentum.py` | Triển khai VGB-Momentum với tham số chi và quy tắc cạnh chung. | Trạng thái momentum và seed được lưu; các trường hợp giới hạn khớp thiết kế đã xác minh. |
| 6.8 | `tests/test_vgb.py` | Kiểm tra cạnh, trọng số, candidate trùng, terminal anchoring và fallback trên toy states. Giải quyết sai kernel mà quality metric khó phát hiện. | Bao phủ lambda=0, zero values và hết budget; không suy ra stationary guarantee từ finite run. |

Cổng kiểm chứng: Các baseline dùng cùng backbone/reward/terminal policy. Ghi rõ các khác biệt upstream được audit, trước khi có kết quả CARS.

**7) Trie, oracle và proposal CARS**

Kiểm chứng xác suất và pruning trước khi ước lượng giá trị. Triển khai leaf rejection trước; chỉ gọi cấu hình full CARS khi có oracle pruning nhánh được kiểm chứng.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 7.1 | `tests/fixtures/toy_grammar.lark` | Định nghĩa grammar nhỏ có support liệt kê được. Giải quyết kiểm chứng pruning không dựa vào parser SMILES phức tạp. | Có prefix còn hoàn tất được, prefix bất khả thi và trường hợp phụ thuộc giới hạn chiều dài. |
| 7.2 | `src/cars_vgb/cars/trie.py` | Lưu raw token probability, child, failure cones và surviving mass trong log-space. Giải quyết đếm trùng nhánh thất bại và underflow. | Unknown child có mass 1; cập nhật không tăng mass; cone trùng/chồng lấp chỉ bị loại một lần; có freeze/serialization contract. |
| 7.3 | `src/cars_vgb/cars/oracle.py` | Cung cấp kết quả certified failure hoặc unknown theo grammar/token/position constraints. Giải quyết prune nhầm vì partial SMILES chưa parse được. | Chỉ loại khi mọi completion có reward 0; với mask có lỗ không ghép token đã biết thành prefix giả; trường hợp chưa chứng minh được trả unknown. |
| 7.4 | `configs/cars.yaml` | Chốt discovery budget, fixed m estimation, rho, oracle, cache và giới hạn bộ nhớ. Giải quyết adaptation hoặc stopping không khai báo. | Tách leaf-only ARS với branch-pruning CARS; số draw/budget được chốt bằng profiling, không lấy tùy ý rồi gắn nhãn paper. |
| 7.5 | `src/cars_vgb/cars/proposal.py` | Sample theo raw probability nhân surviving child mass, trả log proposal và root mass. Giải quyết sampling sau pruning bị lệch khỏi Q_W. | Với W đóng băng có đúng Q_W=q_rho·1_A/B_W; estimation không âm thầm mở rộng W trong lúc lấy draw. |
| 7.6 | `tests/test_trie.py` | Đối chiếu root mass với tổng enumeration, overlap, monotonicity và mass-zero. | Phân biệt certified support exhaustion với numerical underflow. |
| 7.7 | `tests/test_oracle.py` | Kiểm tra oracle trên support toy, prefix, length cap và arbitrary masks. | Không có successful leaf bị prune; partial RDKit failure không được dùng làm chứng nhận. |
| 7.8 | `tests/test_cars_proposal.py` | So sánh xác suất proposal với enumeration và phân phối accepted có điều kiện trên toy. | Kiểm tra trực tiếp xác suất và normalization; kiểm tra sampling nhiều seed chỉ là bằng chứng bổ sung. |

Cổng kiểm chứng: Mass và proposal đúng trước khi dùng CARS cho giá trị. Grammar SMILES phải bao phủ support target; phần chưa có chứng nhận an toàn không được prune.

**8) Estimator CARS, cache và lấy mẫu thích nghi**

CARS đánh giá trạng thái khi VGB truy vấn, không cần giai đoạn train neural toàn cục. Discovery học trie, rồi trie đóng băng để ước lượng bằng các draw mới.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 8.1 | `src/cars_vgb/cars/estimator.py` | Điều phối discovery rồi fixed-m estimation độc lập; tính V_hat=B_W·mean(tau) và interval. Giải quyết nhầm surviving mass với xác suất thành công thật. | Tính cả draw thất bại trước acceptance; không dedup; zero observed success vẫn có upper bound; chỉ nhận query khi reserve đủ discovery và toàn m draw. |
| 8.2 | `src/cars_vgb/cars/cache.py` | Cache full masked state cùng model/tokenizer/reward/rho/estimator metadata; lưu/nạp trie và value records. Giải quyết reuse sai ngữ cảnh và mất trie giữa task. | Giá trị hoàn tất cố định trong run; snapshot chỉ nạp khi metadata khớp; eviction/recompute có cost, không thay đổi target. |
| 8.3 | `src/cars_vgb/verifiers/cars.py` | Adapter CARS estimator qua verifier interface cho batch VGB. Giải quyết tích hợp mà không đổi kernel của VGB. | Giữ thứ tự batch, exact terminal values, uncertainty/cost và unavailable status; cache miss được tính đủ. |
| 8.4 | `src/cars_vgb/sampling/rejection.py` | Triển khai RS, ARS leaf-only và CARS có oracle trên cùng q_rho. Giải quyết so sánh lợi ích pruning mà không đổi backbone hoặc completion policy. | Giữ trie qua các attempt trong cùng ngữ cảnh; ghi attempts, success và chi phí; accepted output không bị dùng để tính trung bình reward estimator. |
| 8.5 | `tests/test_value_estimator.py` | Đối chiếu estimator với giá trị toy chính xác qua nhiều discovery sets và seed. Giải quyết sai số do freeze, acceptance hoặc stopping. | Bao phủ không pruning, B=0, reward toàn 0/1 và zero successes; kiểm tra expectation/coverage với dung sai có cơ sở. |
| 8.6 | `tests/test_cache.py` | Kiểm tra context keys, freeze, eviction và round-trip snapshot. | Không reuse qua checkpoint, threshold, future revealed tokens hoặc rho khác; save/load giữ raw probability và mass. |

Cổng kiểm chứng: Có VGB-CARS-rho và CARS sampling độc lập với training neural. Các draw discovery và estimation có trace/cost riêng; chưa gọi V_rho là V* của native MDM.

**9) Hiệu chỉnh và kiểm chứng V* theo reference rõ ràng**

Hoàn thiện phần chứng minh bằng toy joint trước khi cân nhắc mở rộng importance correction tới QM9.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 9.1 | `src/cars_vgb/verifiers/importance.py` | Dùng w=pi_rho/q_rho để ước lượng tử số bằng CARS và mẫu số bằng q_rho chưa pruning hoặc exact normalizer. Giải quyết q_rho khác joint conditional ở mask có lỗ. | Failed completions bị prune vẫn góp vào mẫu số; báo ESS, weight concentration, uncertainty và scoring cost; không clip weight âm thầm. |
| 9.2 | `tests/test_importance.py` | Đối chiếu likelihood, numerator/denominator, conditional value và support với enumeration. Giải quyết ratio sai nhưng nhìn có vẻ hợp lý. | Prefix rút về estimator đơn giản; mask có lỗ cho thấy target khác nhau; ghi finite-sample ratio bias, không tuyên bố unbiased. |

Cổng kiểm chứng: Nếu QM9 importance variance/cost không khả thi, giữ toy verification và báo giới hạn. Không suy ra native VGB có exact leaf law chỉ vì value đã được hiệu chỉnh.

**10) Run factory, generation và panel kiểm chứng**

Chỉ viết factory đầy đủ sau khi mọi sampler/verifier đã tồn tại; không để nhánh method là placeholder. Các entrypoint Modal tiếp tục chỉ điều phối.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 10.1 | `src/cars_vgb/experiments/run.py` | Ghép model, reward, sampler, verifier, RNG, budget và writer cho một run. Giải quyết so sánh thay đổi âm thầm nhiều thành phần. | Hỗ trợ toàn bộ method đã chốt, target compatibility checks, cold/warm cache labels và immutable run manifest. |
| 10.2 | `modal/generate.py` | Entry point GPU gọi run factory cho một method/config/seed. | Lưu samples, value queries và costs lên Volume; không yêu cầu neural checkpoint cho Base/MC/CARS. |
| 10.3 | `modal/run_generate.sh` | Gọi modal run modal/generate.py::main với method/config/seed. | Chuyển tiếp nguyên vẹn tham số, giữ exit code và không tự chạy nhiều seed. |
| 10.4 | `src/cars_vgb/experiments/value_checks.py` | Tạo panel từ base/VGB snapshots, stratify mask patterns và đánh giá exact/MC/neural/CARS theo cùng target. Giải quyết selection bias và ground truth không rõ. | Panel có nguồn/ID cố định; QM9 high-sample MC được gọi reference có bất định, không gọi exact ground truth. |
| 10.5 | `modal/prepare_value_panel.py` | Entry point tạo panel từ các run nguồn hoặc sampler đã có; dùng GPU khi cần sinh thêm snapshot. | Panel được khóa trước so sánh estimator và có manifest; không cần CARS output để tạo panel baseline đầu tiên. |
| 10.6 | `modal/run_prepare_value_panel.sh` | Gọi modal run modal/prepare_value_panel.py::main. | Nhận source run IDs/panel ID và giữ exit code. |
| 10.7 | `modal/check_values.py` | Entry point chạy enumeration toy hoặc các estimator trên panel QM9. | Kết quả có target/reference ID, uncertainty, ESS nếu áp dụng và tổng chi phí. |
| 10.8 | `modal/run_check_values.sh` | Gọi modal run modal/check_values.py::main. | Phân biệt chế độ toy/CPU và QM9/GPU bằng config, giữ exit code. |

Cổng kiểm chứng: Chạy được một run nhỏ và một panel cố định qua cùng contract. Panel VGB đầu tiên dùng baseline đã có, tránh phụ thuộc vòng vào CARS.

**11) Metrics và tổng hợp kết quả**

Hoàn thiện evaluator trước benchmark lớn để khóa mẫu số và cách tính chi phí, không thay evaluator sau khi nhìn kết quả cuối.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 11.1 | `src/cars_vgb/evaluation/metrics.py` | Tính Pass@95, validity, QED, uniqueness, novelty, diversity và quality/cost từ artifact. Giải quyết chỉ báo cáo valid hoặc accepted samples. | Invalid/timeout/unfinished ở mẫu số yêu cầu; fingerprint settings và chính sách subsampling được ghi; estimator draws không dedup. |
| 11.2 | `src/cars_vgb/evaluation/aggregate.py` | Gộp run theo seed/config, tính uncertainty và kiểm tra provenance tương thích. Giải quyết coi snapshot tương quan là mẫu độc lập. | Tách run thất bại/thiếu artifact rõ ràng, không âm thầm loại; cluster rollout khi cần, cùng threshold/model cho comparison. |
| 11.3 | `src/cars_vgb/evaluation/plots.py` | Tạo quality–cost, value-error–cost và cache cost plots. Giải quyết đánh giá chất lượng mà bỏ qua nested compute. | Trục và chi phí có đơn vị, uncertainty và config ID; không suy FLOPs ratio chỉ từ số tham số. |
| 11.4 | `modal/evaluate.py` | Entry point CPU đọc artifact Volume, tạo report/bảng/đồ thị và hỗ trợ tải artifact được chọn về local. | Tái tính metrics không cần sinh mẫu hoặc load backbone; giữ raw artifact bất biến. |
| 11.5 | `modal/run_evaluate.sh` | Gọi modal run modal/evaluate.py::main. | Nhận comparison/run IDs và tùy chọn tải báo cáo; giữ exit code. |

Cổng kiểm chứng: Có evaluator cố định chạy từ artifact nhỏ, kiểm tra bằng kết quả tính tay trước benchmark.

**12) Kiểm chứng tích hợp và task verify**

Kiểm tra xuyên suốt pipeline với fake backbone/model nhỏ. Không viết test chỉ lặp công thức implementation mà không có đối chứng độc lập.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 12.1 | `tests/test_budget.py` | Đối chiếu counter với số forward thật qua nested MC/CARS, batching, backtrack và fallback. | Discovery, estimation, importance scoring và completion đều được tính; cap/reserve không cho bắt đầu query không thể hoàn tất. |
| 12.2 | `tests/test_pipeline.py` | Chạy pipeline nhỏ với model/fixture, checkpoint round-trip, calibration, sampler, verifier và evaluator. | Bắt lỗi schema/target/batch order, resume state và artifact flow; không tải toàn QM9 hoặc bắt buộc GPU cho kiểm chứng toy. |
| 12.3 | `tests/test_no_resource_dependency.py` | Kiểm tra import, package và smoke pipeline trong môi trường không có resource. Giải quyết runtime vô tình dựa vào source tham khảo. | Không sửa/di chuyển resource thật; kiểm tra image allowlist, source load paths và model config không trỏ resource. |
| 12.4 | `modal/verify.py` | Entry point CPU chạy test suite phù hợp trong image, lưu tests.txt và trạng thái; GPU smoke chỉ khi cấu hình yêu cầu. | Exit code phản ánh test failure, logs persist trên Volume; toy verification không cần checkpoint production. |
| 12.5 | `modal/run_verify.sh` | Gọi modal run modal/verify.py::main. | Nhận test selection/config và giữ nguyên exit code. |

Cổng kiểm chứng: Hoàn tất kiểm chứng xác suất, reward, chi phí và import độc lập trước các lượt chạy lớn. Smoke Modal xác nhận artifact đã commit có thể đọc ở task tiếp theo.

**13) Benchmark có kiểm soát**

Khóa matrix sau pilot về chi phí và trước đánh giá cuối. Phân biệt paper reproduction với thí nghiệm mở rộng của dự án.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 13.1 | `configs/benchmark.yaml` | Chốt method/verifier/target, N, seed, output count, total work cap và cache policy. Giải quyết grid được chọn lại sau khi thấy kết quả. | Mỗi setting có nguồn; không coi pilot 128 hay 3 seed × 1.000 output là protocol paper đã xác minh; các số chưa chốt phải khai báo tường minh. |
| 13.2 | `src/cars_vgb/experiments/benchmark.py` | Điều phối matrix qua run factory, tạo run/attempt IDs và resume các đơn vị chưa hoàn tất. Giải quyết trùng kết quả, ghi đè và retry làm mất chi phí. | Mỗi worker ghi đường dẫn riêng; attempts thất bại giữ trace/cost; output yêu cầu và seed không thay đổi giữa phương pháp. |
| 13.3 | `modal/benchmark.py` | Entry point điều phối trên CPU, cấp tài nguyên theo worker gọi experiments/run.py và giới hạn concurrency. | Không giữ GPU cho coordinator; không import entrypoint từ thư mục modal như package; worker dùng cùng image/config hash và persist completion manifests. |
| 13.4 | `modal/run_benchmark.sh` | Gọi modal run modal/benchmark.py::main với matrix/config và tham số resume. | Không tự sửa matrix hoặc chọn best config; giữ exit code và báo run chưa hoàn tất. |

Cổng kiểm chứng: Đã khóa protocol, estimator budgets và evaluator. Chỉ bắt đầu benchmark cuối khi verification và pilot qua.

**14) Hoàn thiện entrypoint và tài liệu sử dụng**

Đóng gói phần đã hoạt động thành hướng dẫn có thể tái tạo. Đây là các file điều phối/tài liệu cuối cùng vì chúng tham chiếu toàn bộ pipeline.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 14.1 | `src/cars_vgb/__main__.py` | Cung cấp entrypoint local cho các tác vụ dựa trên module đã hoàn thành. Giải quyết dùng lại logic khoa học khi debug toy/local. | Không import resource, không tạo cloud job khi chỉ import hoặc xem help; không có nhánh chưa implement. |
| 14.2 | `README.md` | Hướng dẫn môi trường, model provenance, pipeline Modal, artifact và tái tạo kết quả. | Mọi task/script được mô tả đúng, chỉ báo có kết quả/checkpoint khi artifact thật đã tồn tại. |
| 14.3 | `modal/README.md` | Cập nhật tài liệu hiện có theo CLI, tài nguyên, Volume và thứ tự task thực tế. | 11 cặp Python/shell khớp file đã triển khai; giải thích rõ CARS/MC không cần neural training. |
| 14.4 | `PLAN.md` | Đối chiếu kế hoạch nghiên cứu với implementation cuối, ghi settings/protocol đã xác minh và các sai khác paper. | Không giữ các con số đề xuất ban đầu như setting paper; các quyết định có nguồn và giải thích. |
| 14.5 | `implementation.md` | Đối chiếu cây thư mục, thứ tự file và điều kiện hoàn thành trong tài liệu này với implementation thực tế. | Checklist phản ánh đúng công việc đã làm, chưa làm và giới hạn; không đánh dấu hoàn thành chỉ vì file đã tồn tại. |

Cổng kiểm chứng: Source/config/tests có thể chạy độc lập với resource; mọi task Modal có script tương ứng và hướng dẫn đúng artifact/protocol cuối.

**Thứ tự chạy task sau khi code đã hoàn thành**

Thứ tự dưới đây dành cho thực thi về sau, không phải các lệnh đã chạy trong STEP 5. Mỗi task chỉ được bắt đầu khi artifact đầu vào và kiểm chứng cần thiết đã có.

| Thứ tự chạy | Task hoặc bước setup | Phụ thuộc và kết quả |
| --- | --- | --- |
| 1 | Setup môi trường local/Modal | Cài package/SDK từ môi trường đã chốt, xác thực tài khoản, xác nhận Volume/path và chạy kiểm chứng import local. Credential không lưu trong repository. |
| 2 | verify: toy/CPU | Kiểm chứng xác suất, parser, reward, cache và budget bằng fixtures; không cần assets production. Kiểm chứng phụ thuộc checkpoint chạy ở bước sau. |
| 3 | prepare_assets | Tạo dataset/tokenizer/split manifest trên Volume. |
| 4 | train_base hoặc checkpoint đã xác minh | Train/resume sau smoke nhỏ, hoặc cung cấp checkpoint tương thích có provenance; kiểm tra loader, tokenizer và vocabulary. Chốt một backbone hash chung. |
| 5 | calibrate | Dùng backbone đóng băng và sampler calibration đã khai báo; khóa threshold ID độc lập với output đánh giá. |
| 6 | collect_rollouts | Dùng backbone và threshold đã khóa để tạo native/rho label riêng cho baseline neural. |
| 7 | train_verifier | Huấn luyện neural-native/neural-rho. Chỉ cần cho phương pháp sử dụng neural; CARS/MC có thể bỏ qua bước 6–7. |
| 8 | generate: baseline nhỏ | Chạy Base/BoN và VGB với neural hoặc MC đã có; lấy snapshot baseline cho panel. |
| 9 | prepare_value_panel | Khóa panel từ snapshot base/VGB và các dạng mask; không cần chờ benchmark CARS. |
| 10 | check_values | Chạy exact toy trước, rồi so sánh estimator trên panel với cùng target và budget; kiểm tra importance mode khi phù hợp. |
| 11 | generate: pilot CARS/VGB-CARS | Đo discovery, estimation, cache và tổng forward work; chọn cấu hình bằng pilot riêng trước đánh giá cuối. |
| 12 | verify: tích hợp Modal | Kiểm tra checkpoint thật, artifact giữa task, completion, nested accounting và failure/resume policy sau khi cấu hình pilot được khóa. |
| 13 | benchmark | Chạy matrix/seed/output count đã chốt, giữ trace của mọi attempt và chi phí thực; không tune trên kết quả cuối. |
| 14 | evaluate | Tổng hợp raw artifact thành metrics, uncertainty, figures và report; tải artifact được chọn về local khi cần. |

Chi phí lớn chỉ phát sinh khi các task thực thi về sau; việc tạo tài liệu hiện tại không khởi tạo image, Volume, GPU job hoặc download.

**Artifact được sinh ở đâu và giải quyết việc gì**

| Artifact trong cây thư mục | Module/task sinh | Mục đích |
| --- | --- | --- |
| data/qm9/raw/; splits.json | qm9/dataset.py; prepare_assets | Dataset tải về và danh sách ID split cố định, kiểm tra dữ liệu giữa các run. |
| data/calibration/<id>/samples.jsonl; calibration.json | qm9/calibration.py; calibrate | Output calibration, threshold và provenance; tái kiểm tra quantile mà không sinh lại mẫu. |
| data/rollouts/<id>/states.pt; manifest.json | qm9/rollouts.py; collect_rollouts | Snapshot, terminal label và grouping; training verifier có thể tái tạo split. |
| data/value_panels/<id>/states.jsonl; manifest.json | experiments/value_checks.py; prepare_value_panel | Panel cố định và reference/nguồn trạng thái; mọi estimator được chấm trên cùng đầu vào. |
| model_data/base/<id>/config.json; model.safetensors; tokenizer/; training_state.pt; manifest.json | training/base.py và io.py; train_base | Backbone chung có provenance và đầy đủ trạng thái resume; config không phụ thuộc source resource. |
| model_data/verifier/<id>/verifier.pt; training_state.pt; manifest.json | training/verifier.py và io.py; train_verifier | Learned verifier, target và training state; phân biệt verifier native/rho. |
| cache/tries/<id>/trie.npz; values.jsonl; metadata.json | cars/cache.py | Khôi phục trie/value khi khóa ngữ cảnh khớp; cold/warm cache và chi phí tạo cache được ghi riêng. |
| results/raw/<run_id>/config.yaml; manifest.json | experiments/run.py và io.py | Cấu hình đã resolve, seed, hash, target và trạng thái hoàn tất; tái tạo đúng run. |
| results/raw/<run_id>/samples.jsonl | experiments/run.py | Token/output gốc, canonical identity, reward, completeness và timeout; giữ mẫu số output yêu cầu. |
| results/raw/<run_id>/values.jsonl | experiments/run.py; experiments/value_checks.py | Query giá trị, discovery/estimation trace, uncertainty và target; không trộn accepted samples với estimator draws. |
| results/raw/<run_id>/costs.jsonl | budget.py và io.py | Nested forward work, verifier calls, batch/sequence work, oracle/cache/time; đối chiếu ngân sách thực. |
| results/raw/<run_id>/tests.txt | modal/verify.py | Kết quả kiểm chứng và lỗi nếu có; không giả tạo file này cho run chưa kiểm tra. |
| results/summary/<id>/metrics.csv; metrics.json | evaluation/metrics.py; aggregate.py | Metrics và uncertainty dạng bảng/máy đọc, có provenance comparison. |
| results/summary/<id>/report.md | modal/evaluate.py dùng kết quả evaluation | Kết quả, khác biệt paper và giới hạn; kết luận chỉ dựa trên artifact đã có. |
| results/summary/<id>/figures/quality_cost.png; value_error_cost.png; cache_cost.png | evaluation/plots.py | So sánh chất lượng/độ chính xác theo chi phí và tác động cache. |

**Điều kiện kết thúc IMPLEMENT và chuyển sang kết luận thực nghiệm**

- Có package độc lập, tất cả 11 task Modal cùng script tương ứng và không có runtime dependency vào resource.
- Có backbone/tokenizer, split và calibration với manifest xác minh được; cùng reward/threshold cho các phương pháp trong một protocol.
- Có baseline native, baseline cùng q_rho, neural/MC/CARS verifier và VGB tích hợp qua cùng interface/kernel policy.
- Trie, oracle, proposal, frozen-trie estimator, cache, importance mode và budget đã qua kiểm chứng phù hợp; phần QM9 importance chưa khả thi phải ghi giới hạn rõ.
- Có benchmark theo protocol đã khóa và evaluator tái chạy từ artifact; báo cả lợi ích lẫn trường hợp không có lợi ích.
- Báo cáo giữ phân biệt V_rho, V* của reference đã chỉ định và các giả định còn thiếu cho native MDM-VGB; không tự kế thừa bảo đảm stationary/mixing cho phiên bản chưa thỏa giả định.
