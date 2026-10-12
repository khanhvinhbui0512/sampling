# Thứ tự triển khai CARS và MDM-VGB trên QM9

Tài liệu giữ thứ tự implement của STEP 5 và đối chiếu source/docs sau Step 14. Các bảng nêu nhiệm vụ, vấn đề và điều kiện hoàn thành; source đã viết không đồng nghĩa kiểm chứng đã qua. Lượt Step 14 chỉ viết entrypoint/tài liệu và cập nhật image allowlist, không chạy code, help, test hoặc Modal.

Tên tài liệu chính thức là implementation.md; phần cây thư mục được đặt ngay trong tài liệu này, không tạo thêm một bản implement.md trùng nội dung.

Đọc cùng [PLAN.md](PLAN.md) về phương pháp và [modal/README.md](modal/README.md) về các task Modal. Code tham khảo trong resource chỉ được đọc. Phần cần dùng lại phải được sao chép hợp lệ vào package độc lập, ghi nguồn/license và đổi import; không sửa/import/cài/mount/upload resource trong runtime.

**Phạm vi bắt buộc sau khi rà lại mục tiêu:** (A) đối chiếu MDM-VGB-QM9 và CARS thích nghi cho QM9; (B) nhúng CARS vào verifier của MDM-VGB để xấp xỉ conditional V* theo native reference đã khóa. Step 8 hoàn thiện rho diagnostics; Step 9 phải giải quyết nhánh native trước benchmark chính. Nếu chỉ có VGB-CARS-rho hoặc V* của pi_rho, phần (B) chưa hoàn thành. Source/config/tests native của Step 9 đã được viết và review tĩnh, nhưng chưa chạy hoặc đạt cổng kiểm chứng.

**Quy tắc thực hiện**

- Đi theo thứ tự 1.1 → 1.2 → … → 14.5 trong các bảng. Hoàn thiện chức năng của file trước khi sang file kế tiếp; không dùng pass, NotImplementedError hoặc nhánh giả để gọi là đã xong.
- File chỉ import các module dự án đã hoàn thành trước nó. Các protocol được định nghĩa sớm; factory và entrypoint tổng hợp đặt cuối để tránh phụ thuộc vòng. __init__.py tối thiểu là một file hoàn chỉnh theo đúng nhiệm vụ, không cần re-export mọi module.
- Kiểm tra ngay điều kiện hoàn thành của từng file và cổng kiểm chứng cuối giai đoạn. File test đứng sau module mà nó kiểm tra; chạy kiểm chứng nhỏ trước khi triển khai tiếp phần phụ thuộc.
- Nếu kiểm chứng phát hiện lỗi, sửa file gây lỗi trước khi tiếp tục. Thứ tự tuyến tính không có nghĩa cấm sửa lỗi hoặc cập nhật tài liệu theo bằng chứng.
- Mỗi task Modal có một file modal/<task>.py và một script modal/run_<task>.sh gọi modal run modal/<task>.py::main. Không tạo modal/__init__.py; thuật toán nằm trong src.
- Mọi file modal/<task>.py theo template Nam cung cấp: khai báo trực tiếp App, Image.from_registry với tag nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 và Python 3.11, danh sách pip/file, Volume.from_name và @app.function run_command. Giữ main cho modal run và khối __main__ dùng app.run() cho python; điền số cứng cho CPU/GPU, memory, timeout và Volume ngay trong từng file task.
- Dự án chỉ dùng QM9. Đọc YAML thường thành dictionary; ghi nguồn setting bằng comment và PAPER_CHECK.md. Viết các bước xử lý trực tiếp để dễ đọc, không xây lớp schema/override/inheritance cho nhiều dataset.
- src là package Python gốc, có src/__init__.py; các module nằm trực tiếp trong src. Dùng import đầy đủ như src.qm9, src.cars và src.sampling; types.py và io.py được tham chiếu qua src.types và src.io. pyproject.toml đóng gói src cùng các subpackage; image Modal giữ thư mục src dưới project root. Entry point local là run.py với --task, chạy từ repo root; không dùng lệnh chạy module và không dùng pathlib.
- Phân biệt thứ tự viết file với thứ tự chạy task. CARS-rho đánh giá trạng thái online bằng discovery → frozen-trie estimation → cache; không cần train neural verifier. Neural training là nhánh baseline riêng.
- Target kiểm chứng ban đầu là V_rho theo completion q_rho. Target nhánh tích hợp chính là V* theo native reference được định nghĩa ở PLAN mục 4.3/5.3; không đổi tên V_rho hoặc pi_rho để coi như đã hoàn thành. Source label của upstream neural và control native-conditional phải có nhãn riêng; không giả định trajectory 128-step có cùng conditional law với native inference 32-step.
- Quy mô pilot/benchmark phải có nguồn hoặc được chốt riêng trước đánh giá cuối. Các số 128 và 3 seed × 1.000 output trong đề xuất ban đầu của PLAN chưa phải protocol paper đã được xác minh.

**Cây source và đường dẫn artifact sau Step 14**

Các file source/config/test/docs dưới đây có mục riêng trong thứ tự triển khai. Nội dung resource giữ nguyên và được thu gọn ở cấp repository tham khảo. workspace chứa các đường dẫn artifact được sinh khi chạy; cây chỉ minh họa các file chính, không khẳng định artifact đã tồn tại. Manifest của từng attempt là danh sách đầy đủ file/hash.

```text
sampling/
├── README.md
├── HANDOFF.md                          # Context bàn giao, bằng chứng thực thi và việc còn thiếu
├── PLAN.md
├── PAPER_CHECK.md                      # Đối chiếu phần đã implement với paper/code gốc
├── implementation.md
├── run.py                              # Step 14: chọn một trong 11 runner local hiện có
├── prepare_assets.py                   # Runner local của Step 2, có task Modal tương ứng
├── train_base.py                       # Runner local của Step 3, có task Modal tương ứng
├── calibrate.py                        # Runner local của Step 4, có task Modal tương ứng
├── collect_rollouts.py                 # Runner local Step 5: sinh shard native/rho
├── train_verifier.py                   # Runner local Step 5: train/resume neural verifier
├── generate.py                         # Runner local Step 10: một method/config/seed
├── prepare_value_panel.py              # Runner local Step 10: khóa panel trước so sánh
├── check_values.py                     # Runner local Step 10: toy CPU hoặc panel QM9
├── evaluate.py                         # Runner local Step 11: evaluator CPU từ artifact
├── verify.py                           # Runner local Step 12: suite CPU, GPU opt-in
├── benchmark.py                        # Runner local Step 13: matrix khóa, resume unit chưa xong
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
│   ├── native_conditional_rollouts.yaml # Control cùng V* native; Step 9, chưa chạy
│   ├── mdm_vgb.yaml
│   ├── cars.yaml
│   ├── value_checks.yaml                # Panel/reference budgets chưa profile; toy hữu hạn
│   ├── evaluation.yaml                  # Metrics, Morgan/subsampling và uncertainty cố định
│   ├── verification.yaml                # Test selection, CPU threads, GPU smoke opt-in
│   ├── benchmark.yaml
│   └── reference_checks.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── types.py
│   ├── randomness.py
│   ├── io.py
│   ├── budget.py
│   ├── qm9/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── chemistry.py
│   │   ├── tokenizer.py
│   │   ├── reward.py
│   │   ├── calibration.py
│   │   └── rollouts.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── configuration_udlm.py
│   │   ├── modeling_udlm.py
│   │   └── backbone.py
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── native.py
│   │   ├── completion.py
│   │   ├── reference.py
│   │   ├── native_reference.py          # Luật trajectory native/terminal; Step 9, chưa chạy
│   │   ├── base.py
│   │   ├── bon.py
│   │   ├── rejection.py
│   │   ├── candidates.py
│   │   ├── vgr.py
│   │   ├── vgb.py
│   │   ├── momentum.py
│   │   └── terminal.py
│   ├── cars/
│   │   ├── __init__.py
│   │   ├── trie.py
│   │   ├── oracle.py
│   │   ├── proposal.py
│   │   ├── estimator.py
│   │   ├── native.py                    # CARS theo native trajectory; Step 9, chưa chạy
│   │   └── cache.py
│   ├── verifiers/
│   │   ├── __init__.py
│   │   ├── interface.py
│   │   ├── neural.py
│   │   ├── monte_carlo.py
│   │   ├── cars.py
│   │   └── importance.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── verifier.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run.py
│   │   ├── benchmark.py
│   │   ├── value_checks.py
│   │   ├── toy.py                       # Finite CPU enumeration, không import tests/resource
│   │   └── verification.py              # Pytest + tqdm, logs/status/exit code, readback
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       ├── aggregate.py
│       └── plots.py
├── tests/
│   ├── conftest.py
│   ├── test_qm9_protocol.py
│   ├── test_verifiers.py                # MC/anchoring, nhóm rollout, neural checkpoint/resume
│   ├── test_token_rules.py
│   ├── test_trie.py
│   ├── test_oracle.py
│   ├── test_cars_proposal.py
│   ├── test_value_estimator.py
│   ├── test_importance.py
│   ├── test_native_reference.py
│   ├── test_native_cars.py
│   ├── test_vgb.py
│   ├── test_cache.py
│   ├── test_experiments.py              # Step 10: factory/panel/cache/artifacts, chưa chạy
│   ├── test_evaluation.py               # Step 11: đối chứng tính tay/audit/provenance, chưa chạy
│   ├── test_budget.py
│   ├── test_config_paths.py              # Kiểm tra project-relative paths và symlink
│   ├── test_pipeline.py
│   ├── test_no_resource_dependency.py
│   ├── test_verification.py              # Exit code, failed logs, readback và test selection
│   ├── test_benchmark.py                 # Gate, matrix, resume, duplicate và chi phí retry
│   ├── isolated_smoke.py                 # Helper chỉ chạy trong bản sao không có resource
│   └── fixtures/
│       ├── toy_joint.json
│       └── toy_grammar.lark
└── workspace/                          # Artifact runtime; cùng paths local/Modal
    ├── data/qm9/
    │   ├── <asset_id>/attempts/<attempt_id>/
    │   │   ├── raw/
    │   │   ├── tokenizer/
    │   │   ├── splits.json
    │   │   ├── vocabulary.json
    │   │   ├── token_rules.json
    │   │   ├── train_canonical.jsonl
    │   │   └── manifest.json
    │   ├── calibration/<calibration_id>/attempts/<attempt_id>/
    │   │   ├── sampling_protocol.json
    │   │   ├── samples.jsonl
    │   │   ├── calibration.json
    │   │   └── manifest.json
    │   ├── rollouts/<run_id>/attempts/<attempt_id>/
    │   │   ├── rollouts.pt
    │   │   ├── rollouts.pt.metadata.json
    │   │   └── manifest.json
    │   └── value_panels/<run_id>/attempts/<attempt_id>/
    │       ├── protocol.json
    │       ├── states.jsonl
    │       └── manifest.json
    ├── model_data/qm9/
    │   ├── <base_run_id>/attempts/<attempt_id>/checkpoints/step-<step>/
    │   │   ├── config.json
    │   │   ├── model.safetensors
    │   │   ├── training_state.pt
    │   │   ├── training_state.pt.metadata.json
    │   │   ├── tokenizer/
    │   │   └── manifest.json
    │   └── verifier/<run_id>/attempts/<attempt_id>/checkpoints/epoch-<epoch>/
    │       ├── model.json
    │       ├── model.safetensors
    │       ├── training_state.pt
    │       ├── training_state.pt.metadata.json
    │       └── manifest.json
    ├── cache/
    │   ├── huggingface/
    │   └── tries/<cache_id>/<run_scope_hash>/<state_key>.json
    ├── downloads/evaluation/<comparison_id>/<attempt_id>/
    └── results/
        ├── raw/<generation_run_id>/attempts/<attempt_id>/
        │   ├── protocol.json
        │   ├── execution.json
        │   ├── samples.jsonl
        │   ├── costs.jsonl
        │   ├── traces/<sample_id>.json
        │   ├── value_queries/<sample_id>.json
        │   ├── cache_index.jsonl
        │   └── manifest.json
        ├── raw/qm9-verify/attempts/<attempt_id>/
        │   ├── protocol.json
        │   ├── tests.txt
        │   ├── summary.json
        │   └── manifest.json
        ├── value_checks/<run_id>/attempts/<attempt_id>/
        │   ├── protocol.json
        │   ├── execution.json
        │   ├── values.jsonl
        │   └── manifest.json
        ├── benchmarks/<benchmark_id>/attempts/<attempt_id>/
        │   ├── plan.json
        │   ├── evaluation.yaml            # Snapshot; paths gốc được giữ khi resume
        │   ├── dispatch/
        │   ├── returns/
        │   ├── attempts.jsonl
        │   ├── selected_manifests.json
        │   ├── summary.json
        │   └── manifest.json
        └── summary/<comparison_id>/attempts/<attempt_id>/
            ├── metrics.csv
            ├── metrics.json
            ├── runs.jsonl
            ├── audit.jsonl
            ├── report.md
            ├── manifest.json
            └── figures/
                ├── quality_cost.png
                ├── value_error_cost.png
                └── cache_cost.png
```

Trên Modal, workspace được mount tại /root/workspace trên Volume nam-cars-vgb-qm9-artifacts; local dùng workspace từ repo root và là store riêng. Placeholder <...> chỉ được thay bằng ID thực từ run/manifest, không tự chọn latest. Checkpoint files chỉ có sau khi training ghi completed marker; figures chỉ được tạo khi có cohort phù hợp. Phần mô tả artifact cuối tài liệu nêu writer và mục đích.

**Thứ tự implement từng file**

**1) Chuẩn bị môi trường, package và các hợp đồng dùng chung**

Chốt kiểu dữ liệu và ranh giới module trước khi viết thuật toán. Các __init__.py chỉ khai báo package, không import module chưa tồn tại và không tạo tài nguyên cloud khi import.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 1.1 | `pyproject.toml` | Khai báo package src và các subpackage trực tiếp, phiên bản Python và dependencies tương thích cho Torch, Transformers, RDKit, Modal, dữ liệu, kiểm chứng và biểu đồ. Giải quyết khác biệt môi trường local/Modal. | Cài package và import src cùng dependencies trong môi trường sạch; không cài repository từ resource hoặc thêm riêng thư mục src vào sys.path. |
| 1.2 | `.gitignore` | Bỏ qua checkpoint, dataset, cache, artifact lớn và thông tin xác thực. Giữ source, cấu hình và tài liệu trong Git. | Các đường dẫn artifact và credential được loại đúng, còn source/config/tests vẫn được theo dõi. |
| 1.3 | `THIRD_PARTY_NOTICES.md` | Ghi nguồn, revision/hash, license và các thay đổi của phần code sẽ sao chép. Giải quyết provenance khi tách khỏi repository tham khảo. | Có danh sách nguồn dự kiến; chỉ sao chép phần được phép, viết mới phần chưa xác minh quyền sử dụng. |
| 1.4 | `src/__init__.py` | Khai báo package gốc và metadata tối thiểu; tránh import tạo model hoặc kết nối Modal. | Import src không cần dataset, checkpoint hoặc tài khoản Modal. |
| 1.5 | `src/qm9/__init__.py` | Khai báo package dữ liệu và reward QM9. | Không tải dataset khi import. |
| 1.6 | `src/models/__init__.py` | Khai báo package kiến trúc và adapter backbone. | Không load trọng số khi import. |
| 1.7 | `src/sampling/__init__.py` | Khai báo package sampler, không dựng registry bằng các import chưa tồn tại. | Import độc lập thành công. |
| 1.8 | `src/cars/__init__.py` | Khai báo package trie và estimator CARS. | Không khởi tạo trie toàn cục khi import. |
| 1.9 | `src/verifiers/__init__.py` | Khai báo package các nguồn giá trị trạng thái. | Không yêu cầu neural checkpoint khi import. |
| 1.10 | `src/training/__init__.py` | Khai báo package training. | Import không bắt đầu training. |
| 1.11 | `src/experiments/__init__.py` | Khai báo package điều phối thí nghiệm. | Import không bắt đầu run hoặc benchmark. |
| 1.12 | `src/evaluation/__init__.py` | Khai báo package đánh giá artifact. | Import không tải model hoặc sửa kết quả. |
| 1.13 | `src/types.py` | Định nghĩa MaskedState, SampleResult, ValueEstimate, cost records và protocol cho backbone/sampler. Giải quyết việc các module diễn giải trạng thái hoặc target khác nhau. | Hợp đồng có token IDs, mask/token rules, target/reference ID, trạng thái lỗi và chi phí; hỗ trợ backbone giả mà không cần model thật. |
| 1.14 | `src/config.py` | Đọc một YAML thành dictionary và kiểm tra đường dẫn tương đối từ repo root. | Đường dẫn runtime không trỏ vào resource; không tự suy ngưỡng reward hoặc số seed/output. |
| 1.15 | `src/randomness.py` | Tách RNG cho sampler, discovery, estimation và từng run. Giải quyết việc batching hoặc query lồng làm thay đổi ngẫu nhiên ngoài ý muốn. | Có seed dẫn xuất ổn định và cơ chế lưu/khôi phục RNG. |
| 1.16 | `src/io.py` | Đọc/ghi JSONL, YAML, checkpoint metadata, hash và manifest theo run/attempt ID. Giải quyết ghi đè và artifact chưa hoàn tất. | Ghi hoàn tất có dấu xác nhận; lỗi ghi không biến artifact dở thành artifact hợp lệ. |
| 1.17 | `src/budget.py` | Đếm forward batch thật, sequence work, lượt verifier và chi phí theo giai đoạn; quản lý phần ngân sách dự phòng completion. Giải quyết bỏ sót rollout lồng. | API nhận event từ model thật, không tăng NFE chỉ vì backtrack; hỗ trợ reserve toàn query trước khi chạy. |

Cổng kiểm chứng: Có package độc lập và hợp đồng đủ để các file sau chỉ phụ thuộc phần đã hoàn thành. Chưa cần checkpoint hoặc chạy GPU.

**2) Dữ liệu, reward và task Modal đầu tiên**

Hoàn thiện phần dữ liệu trước khi train model. Task prepare_assets chuẩn bị dataset/tokenizer, không tự tạo ra trọng số pretrained.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 2.1 | `configs/qm9.yaml` | Ghi dataset/tokenizer ID và revision, split, chiều dài 32, token đặc biệt và quy tắc canonicalization trong YAML thường. | Setting có nguồn trong comment/PAPER_CHECK.md; vocabulary lấy từ assets thực, không đoán 35 hoặc 40. |
| 2.2 | `src/qm9/chemistry.py` | Parse SMILES, canonicalize và tính QED bằng RDKit. | Invalid có kết quả rõ ràng; train và output dùng cùng cách chuẩn hóa và phiên bản RDKit. |
| 2.3 | `src/qm9/dataset.py` | Tải QM9/tokenizer; lưu ID split 5% held-out với seed 42, hash và tập train canonical. | Có manifest revision, split IDs và đối chiếu kích thước; không dùng held-out làm prompt generation vô điều kiện. |
| 2.4 | `src/qm9/reward.py` | Tách điểm R=QED hoặc 0 khỏi reward Pass@95. Kiểm tra hoàn chỉnh trước decode, validity và threshold. | MASK còn lại không thể thành công do bị xóa; threshold chưa có được báo là chưa calibration. |
| 2.5 | `prepare_assets.py` | Runner local gọi trực tiếp prepare_qm9_assets với đường dẫn YAML, asset ID và attempt ID. | Chạy từ repo root bằng python prepare_assets.py; in kết quả ngắn, giữ lỗi và attempt riêng. |
| 2.6 | `modal/prepare_assets.py` | Khai báo App, Image, pip/files, Volume và @app.function run_command; điền số cứng cho CPU/memory/timeout và gọi cùng logic dataset preparation. | main hỗ trợ modal run, khối __main__ hỗ trợ python với app.run(); reload/commit Volume; không báo có backbone khi chưa có weights. |
| 2.7 | `modal/run_prepare_assets.sh` | Gọi modal run modal/prepare_assets.py::main từ project root và chuyển tiếp tham số. | Giữ tham số và exit code, không chứa credential. |

Runner local cho cùng task: `prepare_assets.py` gọi trực tiếp logic đã hoàn thành trong `src/qm9/dataset.py`; tương ứng với `modal/prepare_assets.py` và `modal/run_prepare_assets.sh`. Chạy từ repo root bằng `python prepare_assets.py`; không triển khai dispatcher `run.py` của bước 14 ở Step 2.

`src/qm9/tokenizer.py` là dependency của `dataset.py`: đọc ID từ vocabulary công bố, giữ regex SMILES và CLS/SEP, rồi lưu tokenizer chuẩn để các task sau nạp lại. Vocabulary đã có special token nên không cộng thêm offset như custom class cũ. Audit đối chiếu encoding trên từng dòng; khác biệt với upstream được ghi trong `PAPER_CHECK.md`.

Đường dẫn runtime của Step 2 có tiền tố `workspace/`: local lưu dưới repo, còn image Modal đặt project root tại `/root` và mount một Volume tại `/root/workspace`. Cách này giữ cùng đường dẫn tương đối, không để Volume che source/config. Dataset/tokenizer revision `main` chỉ là selector: resolve sang SHA trước khi tải và lưu SHA vào manifest. Chưa xác minh SHA online trong lúc implement; không tự điền một revision đoán.

Cổng kiểm chứng: Khi người dùng cho phép chạy sau review, thiết lập tài khoản Modal và chạy task prepare_assets. Kiểm tra dataset, tokenizer và manifest tồn tại trên Volume trước training. Lượt IMPLEMENT Step 2 chỉ viết file, không chạy code hoặc cổng kiểm chứng này.

**3) Kiến trúc độc lập và huấn luyện backbone**

Dùng kiến trúc QM9 MDLM làm backbone chung. Sao chép phần kiến trúc cần thiết vào package dự án với license/provenance, không load code qua resource.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 3.1 | `src/models/configuration_udlm.py` | Định nghĩa UDLMConfig trong package nội bộ; giữ các trường kiến trúc tương thích checkpoint. Giải quyết phụ thuộc auto_map vào source bên ngoài. | Config lưu/nạp lại được và không trỏ tới resource. |
| 3.2 | `src/models/modeling_udlm.py` | Định nghĩa kiến trúc UDLM phục vụ objective MDLM, sửa relative imports sang package nội bộ. Giải quyết khả năng training/inference độc lập. | Forward/backward trên model nhỏ chạy được; shape, timestep conditioning và tên trọng số tương thích thiết kế đã chọn. |
| 3.3 | `src/models/backbone.py` | Load hoặc khởi tạo model, load tokenizer, kiểm tra vocabulary và expose logits/token log-probabilities theo contract; gắn instrumentation tại forward thật. | Checkpoint round-trip và batching đúng; vocabulary/token mismatch bị từ chối; chi phí của mọi caller được ghi nhận. |
| 3.4 | `tests/conftest.py` | Cung cấp fixture tokenizer/model nhỏ, backbone giả và thư mục artifact tạm. Giải quyết việc kiểm chứng phải tải model 92,4M tham số. | Fixtures độc lập mạng, nhận phân phối toy từ test và không import module chưa implement. |
| 3.5 | `tests/test_token_rules.py` | Kiểm tra MASK/EOS/PAD/UNK, trạng thái hoàn chỉnh, token đã reveal và shape/vocabulary. Giải quyết lỗi chuỗi bị decode thành thành công giả. | Bao phủ mask có lỗ và các trường hợp decode che lỗi completion. |
| 3.6 | `configs/base_training.yaml` | Ghi kiến trúc, objective MDLM, optimizer, schedule, precision, batch và checkpoint policy. Giải quyết training khác nguồn mà không được khai báo. | Các default như seed 1 và 25.000 bước được gắn nhãn code upstream; không tuyên bố là toàn bộ setting paper đã xác minh. |
| 3.7 | `src/training/base.py` | Train/resume MDLM từ split train; lưu weights, optimizer, scheduler, bước training và RNG. Giải quyết thiếu checkpoint inference có provenance. | Một lượt train nhỏ và resume giữ đầy đủ trạng thái; dataset/tokenizer/config hashes đi kèm checkpoint. |
| 3.8 | `modal/train_base.py` | Entry point GPU cho training/resume backbone, chọn checkpoint/output ID và persist checkpoint trên Volume. | Checkpoint đọc được bằng backbone adapter ở task khác; task thất bại không để lại manifest hoàn thành giả. |
| 3.9 | `modal/run_train_base.sh` | Gọi modal run modal/train_base.py::main với config và tham số resume. | Giữ exit code, không tự training lại khi người dùng chỉ yêu cầu resume. |

Runner local cho Step 3: `train_base.py` gọi trực tiếp `src.training.base.train_base`, tương ứng với `modal/train_base.py` và `modal/run_train_base.sh`. Viết runner sau `src/training/base.py`, trước wrapper Modal. Dataset/checkpoint lấy từ manifest đã hoàn thành; không tự tải pretrained hoặc chọn checkpoint ngầm.

Checkpoint nằm tại `workspace/model_data/qm9/<run-id>/attempts/<attempt-id>/checkpoints/step-XXXXXXX/`. Mỗi checkpoint có marker hoàn thành riêng, weights/config/tokenizer và optimizer/scheduler/RNG. Resume giữ nguyên config, assets và môi trường/source đã ghi, tạo attempt mới; checkpoint vẫn dùng được nếu attempt cũ thất bại sau lúc checkpoint hoàn thành. Modal commit sau mỗi checkpoint; GPU/CPU/memory/timeout được điền trực tiếp trong wrapper.

Lượt IMPLEMENT Step 3 chỉ viết kiến trúc, trainer, fixtures/tests, runners và tài liệu; không chạy test, training hoặc job Modal. Sampler/calibration/verifier thuộc Step 4 trở đi. Các điều kiện hoàn thành yêu cầu thực thi ở bảng trên cần được kiểm chứng sau review.

Cổng kiểm chứng: Vượt kiểm chứng nhỏ trước training QM9 thật. Nếu có checkpoint tương thích với provenance xác minh được, dùng checkpoint đó sau kiểm tra loader; mọi phương pháp dùng cùng hash backbone và tokenizer.

**4) Sampler nền, reference và calibration**

Tách native MDLM khỏi completion q_rho. Hoàn thiện raw generation trước khi xác định threshold; calibration không phụ thuộc một reward nhị phân chưa được khóa.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 4.1 | `tests/fixtures/toy_joint.json` | Lưu joint nhỏ, reward và các masked state có thể liệt kê đầy đủ. Giải quyết việc không có ground truth chính xác cho kiểm chứng xác suất. | Tổng xác suất bằng 1; có prefix, mask có lỗ, trạng thái terminal và các completion reward 0/1. |
| 4.2 | `configs/reference_checks.yaml` | Chỉ định V_rho, joint pi_rho, reveal order và cấu hình enumeration/MC/importance. Giải quyết gọi nhầm rollout value là V*. | Mỗi target có reference ID và support assumptions; toy và QM9 được tách rõ. |
| 4.3 | `src/sampling/native.py` | Triển khai schedule và posterior của native MDLM, giữ token đã biết và thu snapshot khi được yêu cầu. Giải quyết tái tạo baseline diffusion. | Đường suy luận khớp công thức tham khảo; số bước và quy tắc completion được ghi rõ, forward cache không làm sai accounting. |
| 4.4 | `src/sampling/completion.py` | Hoàn tất trạng thái theo thứ tự rho cố định, giữ nguyên mọi token đã reveal. Giải quyết completion có phân phối q_rho xác định để so sánh MC/CARS. | Sinh token và tính log q_rho nhất quán; arbitrary masks giữ đúng ngữ cảnh tương lai đã biết. |
| 4.5 | `src/sampling/reference.py` | Định nghĩa joint pi_rho từ root và tính full-sequence likelihood theo prefix gốc. Giải quyết likelihood bị nhầm với q_rho có token tương lai cố định. | Likelihood khớp tổng/enumeration trên toy; không tuyên bố pi_rho là native diffusion joint. |
| 4.6 | `src/sampling/base.py` | Cung cấp Base-native và Base-rho qua cùng contract kết quả và chi phí. Giải quyết baseline thay đổi cả reference mà không ghi nhận. | Output và manifest phân biệt hai sampler; raw generation chạy được trước khi có threshold. |
| 4.7 | `src/qm9/calibration.py` | Sinh calibration độc lập từ base đã đóng băng, tính percentile 95 của R và lưu threshold/provenance. Giải quyết dùng nhầm percentile dataset hoặc dùng tập đánh giá. | Invalid có R=0; lưu sampler, seed, số mẫu, quy tắc quantile và bất định; khóa threshold ID cho các run sau. |
| 4.8 | `tests/test_qm9_protocol.py` | Kiểm tra split cố định, canonical identity, reward và tính độc lập của calibration bằng dữ liệu nhỏ. Giải quyết leakage và mẫu số không thống nhất. | Không cần tải toàn QM9; kiểm tra trực tiếp ID, quantile trên R gồm invalid bằng 0, threshold requirement và invalid/unfinished handling. |
| 4.9 | `modal/calibrate.py` | Entry point GPU gọi calibration và ghi calibration artifact vào Volume. | Threshold gắn đúng model/tokenizer/sampler; không ghi đè calibration đã khóa. |
| 4.10 | `modal/run_calibrate.sh` | Gọi modal run modal/calibrate.py::main và chuyển tiếp tham số calibration. | Có thể chỉ định checkpoint/calibration ID, giữ exit code. |

Runner local Step 4 là `calibrate.py`, viết sau `src/qm9/calibration.py` và trước wrapper Modal. Nó dùng cùng config `configs/reference_checks.yaml` và manifest assets/checkpoint với `modal/calibrate.py`; không triển khai dispatcher/generation CLI của Step 14. Các giá trị mặc định mới cho calibration (10.000 draw, seed 2026, linear quantile, 1.000 bootstrap) là lựa chọn dự án, không gán cho paper.

Native giữ schedule/posterior upstream, thêm final denoise có accounting để loại MASK còn lại ở endpoint hữu hạn; policy ID ghi rõ lựa chọn này. Cache chỉ tái dùng khi cả batch không đổi và model không có time conditioning. q_rho giữ toàn bộ ngữ cảnh đã biết, reveal trái sang phải; pi_rho tính likelihood từ root toàn MASK. Step 4 chỉ cung cấp các phép tính likelihood/log-weight và fixture enumeration, chưa triển khai estimator MC/CARS/importance của các bước sau.

Calibration lưu tại `workspace/data/qm9/calibration/<calibration-id>/attempts/<attempt-id>/`, giữ invalid bằng 0 và không dedup/retry. Mẫu còn MASK làm attempt thất bại; không thay đổi mẫu số ngầm. Quy ước threshold 0 được khóa là vẫn yêu cầu phân tử hợp lệ. Lượt IMPLEMENT này không chạy code/test/Modal; cổng kiểm chứng dưới đây cần thực hiện sau review.

Cổng kiểm chứng: Có Base-native, Base-rho và threshold độc lập. Không mặc định tái sử dụng threshold số trong config upstream khi chưa xác minh nguồn calibration.

**5) Verifier interface, Monte Carlo và baseline neural**

Hoàn thiện các nguồn giá trị đối chứng trước khi thêm CARS. Neural verifier là baseline cần training riêng; MC và CARS không phải đi qua training neural.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 5.1 | `src/verifiers/interface.py` | Chuẩn hóa values_for_states theo batch, target ID, terminal anchoring, uncertainty và cost. Giải quyết các verifier trả giá trị không cùng ý nghĩa. | Giữ đúng thứ tự state đầu vào; terminal trả reward chính xác mà không gọi model verifier. |
| 5.2 | `src/verifiers/monte_carlo.py` | Ước lượng V_rho bằng số completion cố định từ q_rho, kèm interval và tổng chi phí. Giải quyết thiếu baseline trực tiếp cho estimator CARS. | Không dedup draw; không đổi số mẫu theo kết quả; thiếu ngân sách trả trạng thái unavailable để caller fallback. |
| 5.3 | `configs/rollouts.yaml` | Chỉ định native hoặc q_rho, số bước, snapshot và nhóm train/validation. Giải quyết target mismatch giữa label và inference. | Khác biệt rollout 128 bước và inference 32 bước, nếu giữ, được khai báo; số lượng dữ liệu không giả là setting paper. |
| 5.4 | `src/qm9/rollouts.py` | Sinh trajectory, gắn terminal reward cho snapshot, lưu rollout ID và nhóm split. Giải quyết leakage giữa snapshot tương quan. | Tất cả snapshot cùng rollout thuộc một nhóm; label ghi target/policy/threshold ID. |
| 5.5 | `modal/collect_rollouts.py` | Entry point GPU tạo rollout shards và manifest cho verifier. | Shard có ID riêng và completion status; không trộn native và q_rho vào một target không khai báo. |
| 5.6 | `modal/run_collect_rollouts.sh` | Gọi modal run modal/collect_rollouts.py::main. | Chuyển tiếp config/seed/shard params và giữ exit code. |
| 5.7 | `src/verifiers/neural.py` | Định nghĩa neural verifier và adapter load/batch inference. Giải quyết baseline learned process value và kiểm tra target checkpoint. | Kết quả đúng thứ tự, checkpoint metadata đúng target/vocabulary; không hardcode vocabulary 35 hoặc 40. |
| 5.8 | `configs/verifier_training.yaml` | Chốt kiến trúc, MSE objective, optimizer, split và checkpoint policy cho verifier. | Có cấu hình native-label và rho-label tách biệt; hyperparameter có nguồn. |
| 5.9 | `src/training/verifier.py` | Train/validate neural verifier từ snapshot và terminal label; lưu training state để resume. Giải quyết baseline cùng target với CARS/MC. | Validation nhóm theo rollout; checkpoint có target, dữ liệu, threshold và RNG/optimizer state. |
| 5.10 | `modal/train_verifier.py` | Entry point GPU cho training/resume neural verifier. | Checkpoint có thể dùng ở task generation khác; không thay đổi backbone. |
| 5.11 | `modal/run_train_verifier.sh` | Gọi modal run modal/train_verifier.py::main. | Phân biệt training mới và resume; giữ config/target và exit code. |

Runner local tương ứng là `collect_rollouts.py` và `train_verifier.py`; không triển khai dispatcher/generation CLI của Step 14. `tests/test_verifiers.py` viết kiểm tra toy và pipeline offline nhỏ cho các contract Step 5. Lượt IMPLEMENT này chỉ viết, không chạy code/test/job Modal.

Native dùng 128 bước, batch 256 và 5 snapshot theo config upstream; inference/calibration Step 4 dùng 32 bước và có target khác. Snapshot native mang terminal label của trajectory đã sinh, marginalize thời gian vì neural verifier không nhận time. Không gọi target này là luật hoàn tất mới 128 bước từ mọi masked state. Rho dùng trajectory trái sang phải và label binary cùng threshold; các snapshot hiện là prefix, chưa có đảm bảo generalize tới mask có lỗ. MC-rho dùng draw cố định, reserve toàn bộ query và trả unavailable khi không đủ ngân sách. Leaf được anchor reward chính xác ở cả hai adapter.

Config thu 127.190 rollout mỗi policy, theo upstream dùng tất cả train example làm số lượt sinh vô điều kiện; không lấy phân tử dataset để tính label. Shard có rollout ID toàn cục, RNG riêng và completed manifest. Validation 5% theo rollout, split seed 42, rollout seed 1, MC 128 draw/Wilson interval và checkpoint mỗi 10 epoch là lựa chọn dự án. Trainer sigmoid/MSE/AdamW theo nguồn, lưu full optimizer/RNG và resume ở ranh giới epoch; không fallback dùng validation làm train khi thiếu nhóm. Native/rho phải được chọn rõ qua `--policy` và không được trộn target.

Cổng kiểm chứng: Có neural-native, neural-rho và MC-rho với target rõ ràng. Kết luận regression xấp xỉ V* vẫn cần điều kiện reference consistency.

**6) Baseline BoN, VGR, VGB và Momentum**

Hoàn thiện graph và chính sách terminal trước khi thay bằng CARS verifier. Giữ kernel dùng cùng giao diện giá trị để tránh sửa thuật toán theo từng verifier.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 6.1 | `configs/mdm_vgb.yaml` | Ghi L_f, L_b, K, N, lambda, chi, shortlist và chính sách hết budget. Giải quyết default code bị hiểu nhầm là setting mọi thực nghiệm paper. | Có provenance; K là số candidate được lấy mẫu, không bị chuyển thành top-k truncation. |
| 6.2 | `src/sampling/terminal.py` | Xử lý leaf success/failure, backtracking, zero-weight candidate và unguided fallback có reserve. Giải quyết thoát sớm vì validity hoặc hết budget. | Phân biệt valid với Pass@95; paper completion và upstream completion có nhãn riêng; unfinished/timeout được ghi thất bại. |
| 6.3 | `src/sampling/candidates.py` | Xây candidate, multiplicity, depth scaling và trọng số forward/backward từ verifier interface. Giải quyết khác công thức giữa các phiên bản VGB. | Candidate hợp lệ giữ token rules; trọng số và lambda dùng đúng ngữ cảnh, không tự thêm epsilon tạo support mới. |
| 6.4 | `src/sampling/bon.py` | Sinh N completion theo sampler đã chọn và chọn bằng terminal reward/QED theo protocol. | Tính đủ chi phí cả N completion; tie policy và unsuccessful output được ghi rõ. |
| 6.5 | `src/sampling/vgr.py` | Triển khai value-guided resampling trên candidate dùng chung. Giải quyết baseline hướng dẫn tiến không backtrack. | Dùng verifier interface và accounting chung; terminal anchoring đúng. |
| 6.6 | `src/sampling/vgb.py` | Triển khai edge walk với forward/backward, shortlist và budget/terminal policy. Giải quyết baseline MDM-VGB cần đối chiếu trước tích hợp. | Chạy được với neural và MC verifier; không tính backtrack thuần là model forward. |
| 6.7 | `src/sampling/momentum.py` | Triển khai VGB-Momentum với tham số chi và quy tắc cạnh chung. | Trạng thái momentum và seed được lưu; các trường hợp giới hạn khớp thiết kế đã xác minh. |
| 6.8 | `tests/test_vgb.py` | Kiểm tra cạnh, trọng số, candidate trùng, terminal anchoring và fallback trên toy states. Giải quyết sai kernel mà quality metric khó phát hiện. | Bao phủ lambda=0, zero values và hết budget; không suy ra stationary guarantee từ finite run. |

Cổng kiểm chứng: Các baseline dùng cùng backbone/reward/terminal policy. Ghi rõ các khác biệt upstream được audit, trước khi có kết quả CARS.

Ghi chú triển khai Step 6: Đã viết các thư viện/config/test, **chưa chạy**. `sample()` trả `SampleResult`; `run()` trả thêm trace/value records/RNG và hướng Momentum cho writer ở Step 10. Default Momentum giữ stay theo Definition 4.3; `include_stay=false` là convention upstream. BoN default rank binary reward, tie first; QED ranking là option riêng. Baseline dùng một `BaseSampler` completion được chọn rõ (native hoặc rho), reserve trước guidance; giới hạn tổng base sequence 528 = 16×(32+1) đủ reserve native BoN N=16 kể cả final denoise, là lựa chọn dự án, không phải NFE cap của paper. Verifier có `required_work` để reserve cả nhóm query; thiếu budget không đổi draw count hay gán value 0. Timeout được kiểm tra giữa các bước. Task Modal generation ở Step 10 và runner local chung `run.py` ở Step 14, không ở bước này.

**7) Trie, oracle và proposal CARS**

Kiểm chứng xác suất và pruning trước khi ước lượng giá trị. Triển khai leaf rejection trước; chỉ gọi cấu hình full CARS khi có oracle pruning nhánh được kiểm chứng.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 7.1 | `tests/fixtures/toy_grammar.lark` | Định nghĩa grammar nhỏ có support liệt kê được. Giải quyết kiểm chứng pruning không dựa vào parser SMILES phức tạp. | Có prefix còn hoàn tất được, prefix bất khả thi và trường hợp phụ thuộc giới hạn chiều dài. |
| 7.2 | `src/cars/trie.py` | Lưu raw token probability, child, failure cones và surviving mass trong log-space. Giải quyết đếm trùng nhánh thất bại và underflow. | Unknown child có mass 1; cập nhật không tăng mass; cone trùng/chồng lấp chỉ bị loại một lần; có freeze/serialization contract. |
| 7.3 | `src/cars/oracle.py` | Cung cấp kết quả certified failure hoặc unknown theo grammar/token/position constraints. Giải quyết prune nhầm vì partial SMILES chưa parse được. | Chỉ loại khi mọi completion có reward 0; với mask có lỗ không ghép token đã biết thành prefix giả; trường hợp chưa chứng minh được trả unknown. |
| 7.4 | `configs/cars.yaml` | Chốt discovery budget, fixed m estimation, rho, oracle, cache và giới hạn bộ nhớ. Giải quyết adaptation hoặc stopping không khai báo. | Tách leaf-only ARS với branch-pruning CARS; số draw/budget được chốt bằng profiling, không lấy tùy ý rồi gắn nhãn paper. |
| 7.5 | `src/cars/proposal.py` | Sample theo raw probability nhân surviving child mass, trả log proposal và root mass. Giải quyết sampling sau pruning bị lệch khỏi Q_W. | Với W đóng băng có đúng Q_W=q_rho·1_A/B_W; estimation không âm thầm mở rộng W trong lúc lấy draw. |
| 7.6 | `tests/test_trie.py` | Đối chiếu root mass với tổng enumeration, overlap, monotonicity và mass-zero. | Phân biệt certified support exhaustion với numerical underflow. |
| 7.7 | `tests/test_oracle.py` | Kiểm tra oracle trên support toy, prefix, length cap và arbitrary masks. | Không có successful leaf bị prune; partial RDKit failure không được dùng làm chứng nhận. |
| 7.8 | `tests/test_cars_proposal.py` | So sánh xác suất proposal với enumeration và phân phối accepted có điều kiện trên toy. | Kiểm tra trực tiếp xác suất và normalization; kiểm tra sampling nhiều seed chỉ là bằng chứng bổ sung. |

Cổng kiểm chứng: Mass và proposal đúng trước khi dùng CARS cho giá trị. Grammar SMILES phải bao phủ support target; phần chưa có chứng nhận an toàn không được prune.

Ghi chú triển khai Step 7: Đã viết tám file trong bảng và thêm Lark vào dev dependencies; **chưa chạy code hoặc test**. Trie được viết độc lập theo phương trình paper, không sao chép code CARS chưa có license. Path chỉ chứa token điền vào các vị trí MASK ban đầu theo thứ tự trái sang phải, giữ nguyên full context. QM9 default `leaf_only` là ARS; `token_constraints` là oracle CARS chưa đầy đủ, chỉ chứng nhận lỗi token bất khả sửa theo strict decoder của dự án, không dùng grammar functional-group của paper CARS cho QM9. Frozen proposal không cập nhật trie hoặc retry đến khi thành công; chưa có estimator/discovery driver/cache trên đĩa. Discovery budget, fixed m và max_nodes trong config để null chờ profiling, không phải preset benchmark. Các điều kiện bắt buộc trước benchmark được ghi trong `PAPER_CHECK.md`: target chung, baseline có provenance, kernel/reward/decoding chung và budget đủ chạy guidance. Cap 528 ở Step 6 không đủ cho một cold MC query 128×31; không dùng cấu hình đó để kết luận chất lượng MC/CARS. Task Modal không được thêm ở bước thư viện này.

**8) Estimator CARS, cache và lấy mẫu thích nghi**

CARS đánh giá trạng thái khi VGB truy vấn, không cần giai đoạn train neural toàn cục. Discovery học trie, rồi trie đóng băng để ước lượng bằng các draw mới.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 8.1 | `src/cars/estimator.py` | Điều phối discovery rồi fixed-m estimation độc lập; tính V_hat=B_W·mean(tau) và interval. Giải quyết nhầm surviving mass với xác suất thành công thật. | Tính cả draw thất bại trước acceptance; không dedup; zero observed success vẫn có upper bound; chỉ nhận query khi reserve đủ discovery và toàn m draw. |
| 8.2 | `src/cars/cache.py` | Cache full masked state cùng model/tokenizer/reward/rho/estimator metadata; lưu/nạp trie và value records. Giải quyết reuse sai ngữ cảnh và mất trie giữa task. | Giá trị hoàn tất cố định trong run; snapshot chỉ nạp khi metadata khớp; eviction/recompute có cost, không thay đổi target. |
| 8.3 | `src/verifiers/cars.py` | Adapter CARS estimator qua verifier interface cho batch VGB. Giải quyết tích hợp mà không đổi kernel của VGB. | Giữ thứ tự batch, exact terminal values, uncertainty/cost và unavailable status; cache miss được tính đủ. |
| 8.4 | `src/sampling/rejection.py` | Triển khai RS, ARS leaf-only và CARS có oracle trên cùng q_rho. Giải quyết so sánh lợi ích pruning mà không đổi backbone hoặc completion policy. | Giữ trie qua các attempt trong cùng ngữ cảnh; ghi attempts, success và chi phí; accepted output không bị dùng để tính trung bình reward estimator. |
| 8.5 | `tests/test_value_estimator.py` | Đối chiếu estimator với giá trị toy chính xác qua nhiều discovery sets và seed. Giải quyết sai số do freeze, acceptance hoặc stopping. | Bao phủ không pruning, B=0, reward toàn 0/1 và zero successes; kiểm tra expectation/coverage với dung sai có cơ sở. |
| 8.6 | `tests/test_cache.py` | Kiểm tra context keys, freeze, eviction và round-trip snapshot. | Không reuse qua checkpoint, threshold, future revealed tokens hoặc rho khác; save/load giữ raw probability và mass. |

Cổng kiểm chứng: Có CARS sampling và adapter rho độc lập với training neural, dùng được khi run factory được viết ở Step 10. Các draw discovery và estimation có trace/cost riêng. Đây là cổng kiểm chứng trung gian; chưa hoàn thành tích hợp V* native.

Ghi chú triển khai Step 8: Đã viết sáu file trong bảng và fixture reward toy dùng chung; **chưa chạy code/test/Modal**. Discovery lấy một draw theo W hiện tại rồi mới cập nhật failure cones cho draw tiếp theo. ARS chỉ loại sampled leaf reward=0; CARS kiểm tra các child có raw support tại prefix đã đi qua. Estimation dùng RNG độc lập, trie đóng băng và đủ fixed m; interval Hoeffding có điều kiện theo W được nhân với B_W, là lựa chọn của dự án. Capacity dừng learning, không hạ m. Cache giữ value cố định cả run, chỉ eviction trie khỏi RAM; snapshot JSON tùy chọn lưu nguyên raw probabilities, context/source hash và origin cost. Adapter chỉ mang nhãn V_rho, dùng interface và ValueTable hiện có; không đổi native sampler/kernel. Không thêm runner hoặc Modal task ở bước thư viện này. Các count/cap chưa profile vẫn để null trong configs/cars.yaml; Step 9 native và factory Step 10 chưa được implement.

**9) Xấp xỉ V* native và control cùng target — bắt buộc trước benchmark chính**

Hoàn thiện pi_rho/toy như đối chứng trung gian, rồi native reference theo PLAN mục 4.3/5.3. Đi từ enumeration trajectory nhỏ tới profiling QM9. Không cần likelihood marginal của native terminal nếu ước lượng tử/mẫu trực tiếp trên trajectory native; nếu đổi proposal, phải có likelihood ratio và support được kiểm chứng. Không xây framework nhiều dataset hoặc đổi kernel ngoài khi thay verifier.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 9.1 | `configs/reference_checks.yaml` | Bổ sung protocol native-reference rõ ràng, giữ pi_rho diagnostics riêng. Khóa root, schedule, steps và endpoint với baseline được chọn. | Không thay reference bằng tên gọi; upstream/normalized protocol có nhãn riêng; không tự chọn lại threshold khi thay verifier. |
| 9.2 | `src/sampling/native_reference.py` | Mô tả luật trajectory từ native root và terminal joint suy ra; cung cấp raw transition probabilities/log path probability khi cần. | Giữ logits đầu step cho mọi token trong bước song song, MASK self-loop/reveal timing và endpoint đúng native protocol. Proposal có ràng buộc chỉ được dùng với support/weight đã chứng minh. |
| 9.3 | `tests/test_native_reference.py` | Enumeration trajectory của toy native model; đối chiếu terminal joint, mọi C(z), tử/mẫu và conditional value. | Bao phủ future observations, nhiều trajectory tới cùng terminal, zero support và khác biệt với fresh completion từ z. |
| 9.4 | `src/cars/native.py` | Discovery, trie trajectory và frozen CARS proposal cho tử số N_z=P_native(C(z),tau=1); mẫu số D_z=P_native(C(z)) từ draws chưa pruning theo reward. | Không dùng FailureTrie rho như trajectory trie; không loại self-loop hoặc tái query logits giữa token draw cùng step. W chỉ loại cone có E_z·tau=0; lưu log mass, schema/context native riêng và tính toàn bộ chi phí. |
| 9.5 | `tests/test_native_cars.py` | Đối chiếu native trajectory trie/proposal, pruning và ước lượng N_z/D_z với enumeration. | Không prune successful compatible trajectory; failed compatible completions vẫn góp vào D_z; mass-zero khác numerical underflow. |
| 9.6 | `src/verifiers/importance.py` | Tính conditional ratio và uncertainty; giữ pi_rho/q_rho correction như diagnostic riêng. Khi dùng proposal native khác, thêm đúng path likelihood ratio. | Ratio hữu hạn mẫu có thể chệch; D chưa phân giải hoặc ESS thấp trả unavailable/giới hạn, không gán 0; mọi scoring cost được ghi, không clip weight âm thầm. |
| 9.7 | `tests/test_importance.py` | Kiểm tra rho/native ratios, support và target identity bằng enumeration và các seed cố định. | Không đồng nhất V_rho, V* pi_rho và V* native; failed reward leaves không bị xóa khỏi mẫu số. |
| 9.8 | `configs/native_conditional_rollouts.yaml` | Cấu hình control neural cùng target: terminal từ native root, mask pattern chọn độc lập với terminal. | Là control mới, không ghi đè rollout snapshot upstream; số draw/mask/budget được chốt trước đánh giá. |
| 9.9 | `src/qm9/rollouts.py` | Bổ sung label protocol native-conditional qua config, dùng cùng task collect_rollouts local/Modal hiện có. | Giữ cả reward 0/1, không chọn mask dựa trên token/reward; target ID gắn native reference; group split vẫn theo terminal draw. |
| 9.10 | `src/training/verifier.py` | Nhận label protocol/target native-conditional rõ ràng, giữ kiến trúc và objective của verifier. | Loader không coi posthoc mask là diffusion snapshot; checkpoint của upstream-native, rho và native-conditional không bị tráo. Task local/Modal hiện có nhận config/manifest tương ứng. |
| 9.11 | `src/verifiers/monte_carlo.py` | Bổ sung MC-native-conditional từ cùng native reference; giữ MC-rho hiện có riêng. | Estimate N_z/D_z với fixed work và uncertainty; không đổi native completion mới từ z thành conditional root value bằng tên gọi. Mẫu số chưa phân giải trả unavailable. |
| 9.12 | `tests/test_verifiers.py` | Bổ sung kiểm chứng label protocol, độc lập của mask, group split, checkpoint target và MC cho control mới. | Kiểm tra population conditional target bằng toy enumeration; không chỉ kiểm tra tên protocol. |
| 9.13 | `configs/cars.yaml` | Bổ sung block native-reference/trajectory riêng với budget discovery, tử số và mẫu số đã profiling. | Không sửa ngầm block rho; unknown settings vẫn không runnable; không dùng cap 528 cho query không đủ work. |
| 9.14 | `src/cars/cache.py` | Bổ sung namespace/schema cho snapshot trajectory native, giữ cache rho Step 8 riêng. | Không load FailureTrie rho như native trie; full state, native schedule/endpoint, reward, estimator và source version nằm trong key. |
| 9.15 | `src/verifiers/cars.py` | Bổ sung adapter native-conditional sau adapter rho Step 8; trả ValueEstimate qua giao diện candidate builder đang dùng. | Ordered batch, exact leaf anchoring, target/reference IDs và required_work đúng; không đổi backbone, shortlist hoặc outer VGB kernel. |
| 9.16 | `tests/test_value_estimator.py` | Bổ sung đối chứng neural/MC/CARS cùng target native và kiểm tra adapter integration. | Native target mismatch bị phát hiện; rho chỉ chạy khi ghi nhãn diagnostic/surrogate; chưa suy ra stationary-law guarantee. |

Cổng kiểm chứng: Native conditional estimator đã qua enumeration và có profiling/uncertainty trên QM9 trước benchmark chính. Nếu variance/cost chưa khả thi, báo phần mục tiêu còn thiếu; không tự thay bằng V_rho rồi đánh dấu hoàn thành. Đây là thiết kế tích hợp mới, không phải cách hai paper đã triển khai sẵn. Đối chiếu/sửa baseline Step 6 theo PAPER_CHECK trước chạy; không suy ra native VGB có exact leaf law chỉ vì value đúng target.

Ghi chú triển khai Step 9: Đã viết source/config/tests của 16 mục, **chưa chạy code/test/Modal**. `NativeReference` chỉ hỗ trợ protocol có final denoise tường minh, giữ logits đầu step và MASK self-loop. Trie trajectory native có namespace/schema riêng, tính mass bottom-up để không vượt recursion limit với đường đi QM9 dài. CARS loại cone không tương thích quan sát hoặc terminal reward=0, freeze W rồi dùng RNG discovery/numerator/denominator riêng. Mẫu số giữ nguyên native root law, cả reward-zero và duplicate. Interval Hoeffding của N/D kiểm tra denominator lower bound và số hit; ratio ngoài [0,1], D chưa phân giải hoặc underflow đều unavailable, không clip. Cache native lưu cả N/D diagnostics và origin cost; adapter dùng lại ValueTable, không sửa kernel ngoài. Control `native-conditional` dùng task collect/train local và Modal hiện có, mask count đều trên 0..L rồi subset đều, độc lập terminal/reward, split theo terminal group. Kiến trúc/MSE không đổi; upstream snapshots/rho giữ target riêng. Các count/cap chưa profiling để null, vì vậy **cổng enumeration/QM9 feasibility chưa đạt**; chưa implement Step 10 và chưa thể báo mục tiêu nghiên cứu đã hoàn thành.

Review tĩnh Step 9 (2026-10-12): Đã sửa ESS khi bình phương trọng số bị underflow, chặn mất support do xác suất child underflow, kiểm tra đầy đủ reference/calibration khi trie học failure trực tiếp, và duyệt sibling bằng iterator để giảm bộ nhớ tạm. Đã bổ sung regression tests nhưng **chưa chạy**. Công thức N/D vẫn giữ native target đã khóa; chưa có bằng chứng thực nghiệm cho độ chính xác/chi phí. Trước benchmark cần enumeration, uncertainty/availability trên panel QM9, đối chiếu số học serial/batched tại precision đã chọn, và đo wall time toàn query gồm xử lý trie (khác thời gian model/oracle trong CostRecord). Review này chưa sang Step 10.

**10) Run factory, generation và panel kiểm chứng**

Chỉ viết factory đầy đủ sau khi mọi sampler/verifier đã tồn tại; không để nhánh method là placeholder. Các entrypoint Modal tiếp tục chỉ điều phối.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 10.1 | `src/experiments/run.py` | Ghép model, reward, sampler, verifier, RNG, budget và writer cho một run. Giải quyết so sánh thay đổi âm thầm nhiều thành phần. | Phân biệt upstream-reference, rho diagnostic và native-conditional main comparison; kiểm tra target/reference/schema, cold/warm cache và immutable manifest. Không chấp nhận rho bị gắn nhãn native V*. |
| 10.2 | `modal/generate.py` | Entry point GPU gọi run factory cho một method/config/seed. | Lưu samples, value queries và costs lên Volume; không yêu cầu neural checkpoint cho Base/MC/CARS. |
| 10.3 | `modal/run_generate.sh` | Gọi modal run modal/generate.py::main với method/config/seed. | Chuyển tiếp nguyên vẹn tham số, giữ exit code và không tự chạy nhiều seed. |
| 10.4 | `src/experiments/value_checks.py` | Tạo panel từ base/VGB snapshots, stratify mask patterns và đánh giá exact/MC/neural/CARS theo cùng target. Giải quyết selection bias và ground truth không rõ. | Panel chính dùng native conditional value, có root/prefix/mask có lỗ; panel rho riêng. QM9 reference có uncertainty của tử/mẫu/ESS, không gọi exact ground truth. |
| 10.5 | `modal/prepare_value_panel.py` | Entry point tạo panel từ các run nguồn hoặc sampler đã có; dùng GPU khi cần sinh thêm snapshot. | Panel được khóa trước so sánh estimator và có manifest; không cần CARS output để tạo panel baseline đầu tiên. |
| 10.6 | `modal/run_prepare_value_panel.sh` | Gọi modal run modal/prepare_value_panel.py::main. | Nhận source run IDs/panel ID và giữ exit code. |
| 10.7 | `modal/check_values.py` | Entry point chạy enumeration toy hoặc các estimator trên panel QM9. | Kết quả có target/reference ID, uncertainty, ESS nếu áp dụng và tổng chi phí. |
| 10.8 | `modal/run_check_values.sh` | Gọi modal run modal/check_values.py::main. | Phân biệt chế độ toy/CPU và QM9/GPU bằng config, giữ exit code. |

Cổng kiểm chứng: Chạy được một run nhỏ và một panel cố định qua cùng contract. Panel VGB đầu tiên dùng baseline đã có, tránh phụ thuộc vòng vào CARS.

Ghi chú triển khai Step 10: Đã viết tám mục trong bảng, ba runner local tương ứng, `configs/value_checks.yaml`, `src/experiments/toy.py` và regression tests; **chưa chạy code/test/Modal** theo yêu cầu. Factory khóa assets/backbone/tokenizer/calibration và tách native-conditional, rho, upstream snapshot target. Generation có đủ Base/BoN/RS/ARS/CARS/VGR/VGB/Momentum, ghi mọi output cùng per-output result/trace/query/RNG/budget/wall time, provenance và manifest bất biến; MC/CARS không cần checkpoint neural. Estimate unavailable làm sample thất bại rõ ràng; budget refusal giữ completion policy đã khai báo. Cache cold có scope mới, warm đọc snapshot được đăng ký trong completed manifest và giữ origin cost. Panel lấy BASE/baseline walk hoặc root draw độc lập, có root/prefix/hole, không lọc reward và từ chối nguồn CARS/BoN. Check QM9 dùng MC reference độc lập có uncertainty, không gọi exact; toy có enumeration riêng. Modal dùng GPU điền cứng, check toy dùng CPU, không thêm tqdm vào wrapper. Các count/cap chưa profile vẫn null; **cổng chạy nhỏ và cổng Step 9 chưa đạt**. Chưa triển khai Step 11.

**11) Metrics và tổng hợp kết quả**

Hoàn thiện evaluator trước benchmark lớn để khóa mẫu số và cách tính chi phí, không thay evaluator sau khi nhìn kết quả cuối.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 11.1 | `src/evaluation/metrics.py` | Tính Pass@95, validity, QED, uniqueness, novelty, diversity và quality/cost từ artifact. Giải quyết chỉ báo cáo valid hoặc accepted samples. | Invalid/timeout/unfinished ở mẫu số yêu cầu; fingerprint settings và chính sách subsampling được ghi; estimator draws không dedup. |
| 11.2 | `src/evaluation/aggregate.py` | Gộp run theo seed/config, tính uncertainty và kiểm tra provenance tương thích. Giải quyết coi snapshot tương quan là mẫu độc lập. | Tách run thất bại/thiếu artifact rõ ràng, không âm thầm loại; cluster rollout khi cần, cùng threshold/model cho comparison. |
| 11.3 | `src/evaluation/plots.py` | Tạo quality–cost, value-error–cost và cache cost plots. Giải quyết đánh giá chất lượng mà bỏ qua nested compute. | Trục và chi phí có đơn vị, uncertainty và config ID; không suy FLOPs ratio chỉ từ số tham số. |
| 11.4 | `modal/evaluate.py` | Entry point CPU đọc artifact Volume, tạo report/bảng/đồ thị và hỗ trợ tải artifact được chọn về local. | Tái tính metrics không cần sinh mẫu hoặc load backbone; giữ raw artifact bất biến. |
| 11.5 | `modal/run_evaluate.sh` | Gọi modal run modal/evaluate.py::main. | Nhận comparison/run IDs và tùy chọn tải báo cáo; giữ exit code. |
| 11.6 | `evaluate.py`, `configs/evaluation.yaml`, `tests/test_evaluation.py` | Runner local tương đương, khóa metric/fingerprint/subsampling/uncertainty và viết đối chứng tính tay. | Không chạy trong lượt implement; thử nghiệm ở gate sau review. |

Cổng kiểm chứng: Có evaluator cố định chạy từ artifact nhỏ, kiểm tra bằng kết quả tính tay trước benchmark.

Ghi chú triển khai Step 11: Đã viết metrics/aggregate/plots, runner local và cặp Modal CPU, settings evaluator và test đối chứng tính tay; **chưa chạy code/test/Modal**. Metrics generation giữ đủ requested outputs, chấm lại terminal bằng tokenizer/calibration đã khóa và RDKit; QED-valid tách QED-zero-filled. Morgan radius 2/2048 theo paper CARS, ghi rõ code gốc dùng radius 3; subsampling seeded trên valid outputs giữ duplicates. Value checks giữ unavailable/reference uncertainty/N-D-ESS, cluster theo rollout và tách toy exact/QM9 MC. Group seed kiểm tra model/calibration/source/target/config/cache/execution; duplicate seed không gộp. Ledger nested cost không cộng trùng; batch sizes đọc đúng là tập kích thước đã dùng, không phải histogram; adjusted NFE neural cần đo FLOPs. Audit giữ run thiếu/thất bại/corrupt và partial trace cost đã xác minh, báo incomplete comparison. Báo cáo và PNG ghi vào summary attempt mới; Modal có chọn tải report artifacts. Chỉ bổ sung metadata Step 10 (panel path, MC settings, execution/hardware), không đổi sampling. Cổng evaluator và các cổng Step 9–10 còn pending; **chưa triển khai Step 12**.

**12) Kiểm chứng tích hợp và task verify**

Kiểm tra xuyên suốt pipeline với fake backbone/model nhỏ. Không viết test chỉ lặp công thức implementation mà không có đối chứng độc lập.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 12.1 | `tests/test_budget.py` | Đối chiếu counter với số forward thật qua nested MC/CARS, batching, backtrack và fallback. | Discovery, estimation, importance scoring và completion đều được tính; cap/reserve không cho bắt đầu query không thể hoàn tất. |
| 12.2 | `tests/test_pipeline.py` | Chạy pipeline nhỏ với model/fixture, checkpoint round-trip, calibration, sampler, verifier và evaluator. | Bắt lỗi schema/target/batch order, resume state và artifact flow; không tải toàn QM9 hoặc bắt buộc GPU cho kiểm chứng toy. |
| 12.3 | `tests/test_no_resource_dependency.py` | Kiểm tra import, package và smoke pipeline trong môi trường không có resource. Giải quyết runtime vô tình dựa vào source tham khảo. | Không sửa/di chuyển resource thật; kiểm tra image allowlist, source load paths và model config không trỏ resource. |
| 12.4 | `modal/verify.py` | Entry point CPU chạy test suite phù hợp trong image, lưu tests.txt và trạng thái; GPU smoke chỉ khi cấu hình yêu cầu. | Exit code phản ánh test failure, logs persist trên Volume; toy verification không cần checkpoint production. |
| 12.5 | `modal/run_verify.sh` | Gọi modal run modal/verify.py::main. | Nhận test selection/config và giữ nguyên exit code. |

Runner local Step 12 là `verify.py`, gọi `src/experiments/verification.py`; `configs/verification.yaml` chọn toàn bộ `tests` trên CPU mặc định. Thêm `tests/test_verification.py` để kiểm tra failure/exit code/log hash mà không chạy suite đệ quy; `tests/isolated_smoke.py` chỉ được gọi trong bản sao tạm không có resource. Dev dependencies bổ sung setuptools/wheel để kiểm tra wheel thực tế trong bản sao đó.

Ghi chú triển khai Step 12: Đã viết, **chưa chạy code, test hoặc Modal**. Counter được đối chiếu với network calls độc lập và hook model nhỏ; cache raw logits không bị tính thành forward mới. Pipeline dùng fixture ba dòng, train hai bước/resume, calibration bốn BASE draws và generation nhỏ; các số này chỉ phục vụ test, không đổi setting production/paper. Suite hiện có tiếp tục kiểm tra neural training, xác suất, pruning và các target. Mỗi verification attempt lưu protocol/source hashes, `tests.txt`, `summary.json`; pytest failure giữ manifest failed và mã thoát. Modal mặc định CPU, commit cả log lỗi rồi gọi function CPU khác để kiểm tra hash sau reload; CUDA smoke chỉ bật qua config/CLI. Chưa có bằng chứng chạy readback, chưa đánh dấu gate hay triển khai Step 13.

Cổng kiểm chứng: Hoàn tất kiểm chứng xác suất, reward, chi phí và import độc lập trước các lượt chạy lớn. Smoke Modal xác nhận artifact đã commit có thể đọc ở task tiếp theo.

**13) Benchmark có kiểm soát**

Khóa matrix sau pilot về chi phí và trước đánh giá cuối. Phân biệt paper reproduction với thí nghiệm mở rộng của dự án.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 13.1 | `configs/benchmark.yaml` | Chốt hai deliverable chính: standalone CARS-QM9 vs MDM-VGB-QM9 và thay verifier native-conditional bằng CARS; giữ rho diagnostics riêng. Khóa N, seed, output count, total work cap và cache. | Baseline được đối chiếu upstream, native target qua Step 9 và budget đủ chạy guidance; mỗi setting có nguồn. Không coi pilot 128 hay 3 seed × 1.000 output là protocol paper đã xác minh. |
| 13.2 | `src/experiments/benchmark.py` | Điều phối matrix qua run factory, tạo run/attempt IDs và resume các đơn vị chưa hoàn tất. Giải quyết trùng kết quả, ghi đè và retry làm mất chi phí. | Mỗi worker ghi đường dẫn riêng; attempts thất bại giữ trace/cost; output yêu cầu và seed không thay đổi giữa phương pháp. |
| 13.3 | `modal/benchmark.py` | Entry point điều phối trên CPU, cấp tài nguyên theo worker gọi experiments/run.py và giới hạn concurrency. | Không giữ GPU cho coordinator; không import entrypoint từ thư mục modal như package; worker dùng cùng image/config hash và persist completion manifests. |
| 13.4 | `modal/run_benchmark.sh` | Gọi modal run modal/benchmark.py::main với matrix/config và tham số resume. | Không tự sửa matrix hoặc chọn best config; giữ exit code và báo run chưa hoàn tất. |

Runner local bổ sung cho Step 13 là `benchmark.py`, gọi cùng coordinator `src/experiments/benchmark.py`; `tests/test_benchmark.py` dùng worker tạo artifact nhỏ để kiểm tra gates, resume, giữ partial cost và từ chối completed attempts trùng. Image `modal/verify.py` thêm runner benchmark vào allowlist để verification sau Step 13 vẫn có đầy đủ file, không thêm task hoặc thuật toán khác.

Ghi chú triển khai Step 13: Đã viết, **chưa chạy code, test, pilot hoặc Modal**. `configs/benchmark.yaml` có matrix standalone/integration/rho diagnostic và job kiểm tra MC/neural/CARS trên cùng panel native theo seed; `locked=false`, seeds/output counts và input/gate manifests chưa chốt vẫn để null. N/L_f/L_b/K/lambda lấy từ inference config đã đối chiếu upstream; n*N walk cap tách khỏi hard total work cap. Không tự chọn 128 hay 3×1.000 output.

Mỗi coordinator có plan/config snapshots và dispatch intent bất biến; mỗi worker có run/attempt riêng. Resume cần manifest cụ thể và giữ nguyên source/config/phase/seed/count/input; chỉ skip một completed artifact đã được đối chiếu job, không chọn best/latest khi trùng. `attempts.jsonl` giữ các attempt lỗi/đang dở/không có manifest, chi phí trace kiểm chứng được là lower bound; value-job cost có cả reference MC và được ghi nhãn riêng. Unit thành công không lọc output failed/invalid. `selected_manifests.json` dành cho evaluator Step 11; không gộp retry cost vào metric selected-output mà không ghi rõ.

Pilot readiness là kiểm tra cấu trúc: mỗi generation row có output hoàn chỉnh, mỗi guided row có nonterminal guidance khả dụng, và mỗi seed có common nonterminal value pair. Đây không phải kiểm định value accuracy. Final cần full verification hiện tại, native gate, CARS profiled/ready flags và pilot completed/ready cùng scientific contract; số seed/output giữa pilot và final có thể khác theo config. Coordinator Modal CPU 2/8 GiB, worker RTX-PRO-6000:1, CPU 4/32 GiB, timeout 24h và concurrency 1 được điền trực tiếp. Commit plan/dispatch trước worker, commit cả lỗi rồi reload sau worker. Step 14 chưa triển khai; chưa gate nào được đánh dấu passed bởi lượt viết này.

Cổng kiểm chứng: Đã khóa protocol, estimator budgets và evaluator. Chỉ bắt đầu benchmark cuối khi verification và pilot qua.

**14) Hoàn thiện entrypoint và tài liệu sử dụng**

Đóng gói phần đã hoạt động thành hướng dẫn có thể tái tạo. Đây là các file điều phối/tài liệu cuối cùng vì chúng tham chiếu toàn bộ pipeline.

| Thứ tự | File | Chức năng và vấn đề giải quyết | Điều kiện hoàn thành |
| --- | --- | --- | --- |
| 14.1 | `run.py` | Cung cấp entrypoint local python run.py --task <task> từ repo root cho các tác vụ dựa trên module đã hoàn thành, mỗi task có entrypoint Modal tương ứng. Giải quyết dùng lại logic khoa học khi debug toy/local. | Không import resource, không tạo cloud job khi chỉ import hoặc xem help; in kết quả ngắn và log tiến trình bằng tqdm; không có nhánh chưa implement. |
| 14.2 | `README.md` | Hướng dẫn môi trường, model provenance, pipeline Modal, artifact và tái tạo kết quả. | Mọi task/script được mô tả đúng, chỉ báo có kết quả/checkpoint khi artifact thật đã tồn tại. |
| 14.3 | `modal/README.md` | Cập nhật tài liệu hiện có theo CLI, tài nguyên, Volume và thứ tự task thực tế. | 11 cặp Python/shell khớp file đã triển khai; giải thích rõ CARS/MC không cần neural training. |
| 14.4 | `PLAN.md` | Đối chiếu kế hoạch nghiên cứu với implementation cuối, ghi settings/protocol đã xác minh và các sai khác paper. | Không giữ các con số đề xuất ban đầu như setting paper; các quyết định có nguồn và giải thích. |
| 14.5 | `implementation.md` | Đối chiếu cây thư mục, thứ tự file và điều kiện hoàn thành trong tài liệu này với implementation thực tế. | Checklist phản ánh đúng công việc đã làm, chưa làm và giới hạn; không đánh dấu hoàn thành chỉ vì file đã tồn tại. |

Ghi chú triển khai Step 14: `run.py` dùng stdlib argparse/subprocess với danh sách cố định 11 task, chuyển tiếp flags sang runner local bằng interpreter Miniconda hiện tại và giữ stdout/stderr/tqdm/mã thoát. Import không chạy task; global help không import model/Modal, task help dừng ở parser của runner. Mọi task có Modal pair đã liệt kê, không thêm task khoa học mới. `modal/verify.py` và `modal/benchmark.py` thêm root run.py vào image allowlist để snapshot source khớp local/worker. Các sections README đã hoàn thành giữ nguyên; mục 14 bổ sung entrypoint, provenance, thứ tự thực thi và giới hạn.

Script Modal calibration hiện có `-d`: mã thoát shell phản ánh submission, còn completion/failure cần xem job và manifest remote. Tài liệu mục 14 ghi rõ điều này; không đổi cách chạy task ở bước chỉ hoàn thiện entrypoint/docs.

| Checklist đối chiếu | Bằng chứng hiện có / còn thiếu |
| --- | --- |
| Entrypoint và 11 Modal pairs | Source đã viết, đối chiếu filenames/CLI tĩnh. Chưa chạy import, help hoặc dispatch. |
| Tài liệu và artifact paths | README mục 14, modal/README mục 14, PLAN và cây/table trong tài liệu này đã cập nhật theo writer/config. Không tạo hoặc xác nhận artifact mới. |
| Chuẩn bị QM9 | Bằng chứng chuẩn bị trước đó được ghi trong README mục 2/PAPER_CHECK.md; không chạy lại ở Step 14. |
| Backbone cuối và calibration | Đã ghi nhận checkpoint 5.000/10.000 bước trước mất kết nối; 25.000-step completion chưa xác minh. Calibration chưa chạy, chưa có threshold/manifest để benchmark. |
| Native/estimator/evaluator/Modal gates | Source/test đã viết; full suite, native feasibility, readback và numeric validation chưa chạy. Không chuyển cờ gate sang passed. |
| Lỗi nối artifact Step 13 cần review | `src/experiments/benchmark.py` đọc calibration `protocol.json`, writer `src/qm9/calibration.py` ghi `sampling_protocol.json`. Cần sửa reader theo artifact thật trước benchmark; lượt chỉ Step 14 này ghi nhận và giữ nguyên logic Step 13. |
| Pilot/final và kết luận khoa học | Seeds/counts/manifests còn unset, benchmark vẫn unlocked. Chưa có pilot hoặc kết quả so sánh; rho/toy không thay thế V* native. |

Lượt Step 14 **không chạy code, help, test, build image hoặc Modal**. Các điều kiện gate là việc phải thực thi sau review, không phải checklist đã pass từ việc viết file. Dependency lock, importance QM9, order sensitivity, distillation và editing chưa được chứng minh/triển khai thành thí nghiệm hoàn chỉnh.

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
| 7 | train_verifier | Huấn luyện upstream native-snapshot và matched native-conditional controls riêng; rho là diagnostic. CARS/MC riêng lẻ có thể bỏ qua bước 6–7; full matrix cần hai neural controls. |
| 8 | generate: baseline nhỏ | Chạy Base/BoN và VGB với neural hoặc MC đã có; lấy snapshot baseline cho panel. |
| 9 | prepare_value_panel | Khóa panel từ snapshot base/VGB và các dạng mask; không cần chờ benchmark CARS. |
| 10 | check_values | Chạy exact toy trước, rồi so sánh estimator trên panel với cùng target và budget, giữ independent MC reference có bất định. Importance joint rho có thư viện/toy checks; chưa có QM9 benchmark importance. |
| 11 | generate: pilot CARS/VGB-CARS | Đo discovery, estimation, cache và tổng forward work; chọn cấu hình bằng pilot riêng trước đánh giá cuối. |
| 12 | verify và đối chiếu tích hợp | Chạy lại full fixture suite trên source hiện tại và Modal readback; kiểm tra production manifests qua task tương ứng. verify không tự tải checkpoint production hoặc chạy benchmark để chứng minh feasibility. |
| 13 | benchmark | Chạy matrix/seed/output count đã chốt, giữ trace của mọi attempt và chi phí thực; không tune trên kết quả cuối. |
| 14 | evaluate | Tổng hợp raw artifact thành metrics, uncertainty, figures và report; tải artifact được chọn về local khi cần. |

Chi phí lớn chỉ phát sinh khi các task thực thi về sau; việc tạo tài liệu hiện tại không khởi tạo image, Volume, GPU job hoặc download.

**Artifact được sinh ở đâu và giải quyết việc gì**

| Artifact trong cây thư mục | Module/task sinh | Mục đích |
| --- | --- | --- |
| workspace/data/qm9/<asset_id>/attempts/<attempt_id>/raw/; tokenizer/; splits.json; vocabulary.json; token_rules.json | src/qm9/dataset.py; prepare_assets | Dataset/tokenizer và split/ID cố định; audit và train_canonical.jsonl phục vụ kiểm tra/novelty. |
| workspace/data/qm9/calibration/<id>/attempts/<attempt_id>/sampling_protocol.json; samples.jsonl; calibration.json | src/qm9/calibration.py; calibrate | BASE output, threshold và provenance; tái kiểm tra quantile mà không sinh lại. Writer không tạo artifact tên protocol.json. |
| workspace/data/qm9/rollouts/<id>/attempts/<attempt_id>/rollouts.pt; rollouts.pt.metadata.json; manifest.json | src/qm9/rollouts.py; collect_rollouts | State, terminal label và rollout groups; phân biệt native snapshot/rho/native-conditional independent masks. |
| workspace/data/qm9/value_panels/<id>/attempts/<attempt_id>/states.jsonl; protocol.json; manifest.json | src/experiments/value_checks.py; prepare_value_panel | Panel và reference/nguồn đã khóa trước khi so estimator. |
| workspace/model_data/qm9/<id>/attempts/<attempt_id>/checkpoints/step-<step>/ | src/training/base.py; train_base | config.json, model.safetensors, tokenizer, training_state.pt/metadata và completed checkpoint manifest cho inference/resume. |
| workspace/model_data/qm9/verifier/<id>/attempts/<attempt_id>/checkpoints/epoch-<epoch>/ | src/training/verifier.py; train_verifier | model.safetensors, model.json, training_state.pt/metadata và target/checkpoint manifest; không có verifier.pt của upstream. |
| workspace/cache/tries/<cache_id>/<run_scope_hash>/<state_key>.json | src/cars/cache.py | Snapshot trie/value/metadata/checksum/origin cost khi bật persistent_cache; run scope gồm run/attempt. Cold/warm provenance được ghi riêng. |
| workspace/results/raw/<run_id>/attempts/<attempt_id>/protocol.json; execution.json; manifest.json | src/experiments/run.py và src/io.py | Config, seed, identities, source/environment và completion marker của generation. |
| workspace/results/raw/<run_id>/attempts/<attempt_id>/samples.jsonl; costs.jsonl; traces/ | src/experiments/run.py và src/budget.py | Mọi output yêu cầu, trạng thái và nested forward cost theo phase; trace giữ cả fallback/thất bại. |
| workspace/results/raw/<run_id>/attempts/<attempt_id>/value_queries/; cache_index.jsonl | src/experiments/run.py | Query diagnostics và registered cache provenance; không dùng estimator draws làm output molecules. |
| workspace/results/value_checks/<run_id>/attempts/<attempt_id>/values.jsonl | src/experiments/value_checks.py | Mọi panel state, estimate/reference/uncertainty/query budgets; independent MC reference có chi phí riêng. |
| workspace/results/raw/<run_id>/attempts/<attempt_id>/tests.txt; protocol.json; summary.json | src/experiments/verification.py; verify | Log/status/selection/source hashes thực khi chạy; Modal readback kiểm tra committed artifacts. |
| workspace/results/benchmarks/<id>/attempts/<attempt_id>/plan.json; attempts.jsonl; selected_manifests.json; summary.json | src/experiments/benchmark.py | Matrix/snapshots, dispatch history, immutable retry ledger, explicit worker selections và completion/readiness. |
| workspace/results/summary/<id>/attempts/<attempt_id>/metrics.csv; metrics.json; runs.jsonl; audit.jsonl; report.md | src/evaluation/metrics.py và aggregate.py; evaluate | Metrics/uncertainty/provenance và audit failed/duplicate attempts; không tự sinh molecule. |
| workspace/results/summary/<id>/attempts/<attempt_id>/figures/ | src/evaluation/plots.py | quality_cost.png, value_error_cost.png và cache_cost.png khi cohort/availability phù hợp. |
| workspace/downloads/evaluation/<id>/<attempt_id>/ | modal/evaluate.py | Chỉ report artifacts được yêu cầu và manifest, kiểm tra hash/không ghi đè; không phải bản sao toàn bộ remote workspace. |

**Điều kiện kết thúc IMPLEMENT và chuyển sang kết luận thực nghiệm**

- Có package độc lập, tất cả 11 task Modal cùng script tương ứng và không có runtime dependency vào resource.
- Có backbone/tokenizer, split và calibration với manifest xác minh được; cùng reward/threshold cho các phương pháp trong một protocol.
- Có baseline native, baseline cùng q_rho, neural/MC/CARS verifier và VGB tích hợp qua cùng interface/kernel policy.
- Trie, oracle, proposal, frozen-trie estimator, cache, importance mode và budget đã qua kiểm chứng phù hợp; phần QM9 importance chưa khả thi phải ghi giới hạn rõ.
- Có benchmark theo protocol đã khóa và evaluator tái chạy từ artifact; báo cả lợi ích lẫn trường hợp không có lợi ích.
- Báo cáo giữ phân biệt V_rho, V* của reference đã chỉ định và các giả định còn thiếu cho native MDM-VGB; không tự kế thừa bảo đảm stationary/mixing cho phiên bản chưa thỏa giả định.
