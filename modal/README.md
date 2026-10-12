# Tổ chức các task chạy trên Modal

Modal là môi trường thực thi của dự án. Task prepare_assets, train_base, calibrate, collect_rollouts và train_verifier đã có entrypoint và script trong STEP 6, Step 2–5; Step 10 thêm generate, prepare_value_panel và check_values; Step 11 thêm evaluate; Step 12 thêm verify; Step 13 thêm benchmark. prepare_assets đã chạy thành công ngày 2026-10-12; các task Step 10–13 chưa chạy. Xem manifest và kết quả ở [README section 2](../README.md#2-prepare-qm9-assets).

Mỗi task có một file Python và một script shell tương ứng. File Python cung cấp entrypoint main để nhận tham số và gọi package src; script gọi modal run modal/<task>.py::main. Cú pháp file::entrypoint được mô tả trong [Modal CLI](https://modal.com/docs/cli/latest/run).

Wrapper Modal chỉ in kết quả ngắn; tqdm nằm trong code xử lý ở `src/` và các runner local. Không thêm thanh tqdm bọc ngoài lời gọi remote.

Step 6 bổ sung thư viện BoN/VGR/VGB/Momentum, config và test toy; chưa thêm task chạy. Cặp `modal/generate.py` / `modal/run_generate.sh` thuộc Step 10; runner local chung `run.py` thuộc Step 14 trong `implementation.md`. Các baseline đã có tqdm và kết quả ngắn trong `src/sampling/`; chưa chạy code hoặc test ở Step 6.

Step 7 bổ sung thư viện trie/oracle/frozen proposal CARS và test, chưa có estimator hoặc task mới. Không thêm wrapper Modal cho thư viện; cặp task generate vẫn thuộc Step 10. tqdm nằm trong proposal, có thể tắt khi task bên ngoài đã hiển thị progress. Budget CARS chưa profiling và chưa được chốt; không chạy benchmark từ config Step 7. Code và test Step 7 chưa được chạy.

Phạm vi kiểm chứng chính gồm CARS-QM9 độc lập và CARS thay verifier trong MDM-VGB theo native conditional target. Rho/toy là kiểm chứng trung gian. Step 9 trong `implementation.md` đã viết native-reference estimator và control neural cùng target, nhưng chưa chạy kiểm chứng. Dùng lại task collect_rollouts/train_verifier/generate/check_values với config/manifest có protocol rõ ràng; không thêm task Modal chỉ để đổi reference. Không coi benchmark rho thành hoàn thành mục tiêu V* native.

```text
modal/
├── README.md
├── prepare_assets.py
├── run_prepare_assets.sh
├── train_base.py
├── run_train_base.sh
├── calibrate.py
├── run_calibrate.sh
├── collect_rollouts.py
├── run_collect_rollouts.sh
├── train_verifier.py
├── run_train_verifier.sh
├── prepare_value_panel.py
├── run_prepare_value_panel.sh
├── generate.py
├── run_generate.sh
├── check_values.py
├── run_check_values.sh
├── benchmark.py
├── run_benchmark.sh
├── evaluate.py
├── run_evaluate.sh
├── verify.py
└── run_verify.sh
```

| File Python | Script tương ứng | Task và đầu ra |
| --- | --- | --- |
| prepare_assets.py | run_prepare_assets.sh | Chuẩn bị QM9, tokenizer, split ID và manifest; kiểm tra token/vocabulary. Chưa tạo trọng số pretrained. |
| train_base.py | run_train_base.sh | Huấn luyện hoặc tiếp tục huấn luyện MDLM QM9; lưu checkpoint, optimizer state, RNG state và provenance. Nếu đã có checkpoint tương thích được xác minh, không cần chạy lại task này. |
| calibrate.py | run_calibrate.sh | Sinh tập calibration độc lập từ backbone đã đóng băng; tính và lưu threshold QED cùng sampler, seed và manifest. |
| collect_rollouts.py | run_collect_rollouts.sh | Sinh trajectory/snapshot và label terminal reward cho neural verifier; phân biệt native rollout với q_rho, nhóm split theo rollout. |
| train_verifier.py | run_train_verifier.sh | Huấn luyện neural verifier từ rollout đã chọn, lưu checkpoint và target definition. CARS online không cần bước training này. |
| prepare_value_panel.py | run_prepare_value_panel.sh | Tạo panel trạng thái cố định từ base/VGB, gồm mask có lỗ; lưu nguồn trạng thái và manifest để các estimator dùng cùng đầu vào. |
| generate.py | run_generate.sh | Chạy một cấu hình và seed cho Base, BoN, VGR, VGB, Momentum, RS, ARS hoặc CARS thích nghi; chọn verifier theo config. Lưu output, value queries và chi phí thực. |
| check_values.py | run_check_values.sh | Kiểm chứng giá trị bằng enumeration trên toy joint và panel QM9; so sánh MC/CARS/neural với cùng target, ghi sai số, bất định và chi phí. Chỉ gọi là kiểm chứng V* khi reference được xác định rõ. |
| benchmark.py | run_benchmark.sh | Điều phối generate theo method, verifier, budget và seed trong ma trận đã khóa; lưu manifest từng run và trạng thái hoàn thành. Không tự chọn lại siêu tham số dựa trên kết quả cuối. |
| evaluate.py | run_evaluate.sh | Đọc artifact đã lưu, tính metrics, tổng hợp seed và tạo bảng/đồ thị; hỗ trợ tải báo cáo và artifact được chọn về results local. |
| verify.py | run_verify.sh | Chạy các kiểm chứng xác suất, pruning, reward, cache, budget, tích hợp và khả năng chạy khi không có resource; lưu kết quả tests. |

Task ở đây là một công đoạn pipeline. Các phương pháp sampling dùng chung generate.py với cấu hình khác nhau, để giữ cùng reward, backbone và cách tính chi phí.

Mọi script xác định project root, chuyển tiếp tham số người dùng và trả đúng exit code của Modal. Mỗi entrypoint đọc YAML thường và ghi settings cùng run ID. benchmark chỉ điều phối; GPU được cấp cho worker cần model. prepare_assets, evaluate và kiểm chứng toy dùng CPU; training và generation dùng GPU. Điền trực tiếp loại GPU, CPU, memory, timeout và concurrency trong file Python của từng task.

Mọi file modal/<task>.py theo template Nam cung cấp: import App, Image, Volume từ modal; khai báo app, image, dependency và các file cần đưa vào image ngay trong file; tạo Volume.from_name; định nghĩa run_command bằng @app.function. Image dùng tag bắt buộc nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 với add_python="3.11", theo [Modal Images](https://modal.com/docs/sdk/py/latest/Image). prepare_assets điền trực tiếp gpu=None, cpu=2, memory=8192, timeout=3600, max_containers=1 và retries=0.

Mỗi file giữ @app.local_entrypoint main cho script modal run, đồng thời có if __name__ == "__main__" dùng app.run() để gọi cùng hàm remote khi chạy bằng python. prepare_assets dùng serialized=True và include_source=False để gửi wrapper với cả hai cách gọi; source/config được đưa vào image bằng danh sách khai báo trực tiếp.

train_base dùng cùng template và điền trực tiếp gpu="RTX-PRO-6000:1", cpu=4, memory=32768, timeout=3600*24, max_containers=1, retries=0. Wrapper gọi `src.training.base.train_base`, commit Volume sau mỗi checkpoint và khi kết thúc hoặc thất bại. `--assets-manifest` bắt buộc; `--resume-manifest` chọn checkpoint trung gian có marker hoàn thành, tạo attempt mới và giữ lịch training cũ. Cả hai đường dẫn phải nằm dưới workspace/. Runner local tương ứng là `python train_base.py`; lệnh đầy đủ ở [README section 3](../README.md#3-train-the-qm9-backbone).

calibrate gọi `src.qm9.calibration.calibrate_qm9` với `--assets-manifest` và `--checkpoint-manifest` bắt buộc, config `configs/reference_checks.yaml`. GPU="RTX-PRO-6000:1", cpu=4, memory=32768, timeout=3600*6, max_containers=1, retries=0 được điền trực tiếp. Wrapper reload Volume trước khi đọc và commit khi thành công/thất bại. Lưu draw độc lập, threshold QED, bootstrap, policy/seed/RNG và completed manifest; không có neural verifier ở task này. Runner local là `python calibrate.py`; xem [README section 4](../README.md#4-base-sampling-and-pass95-calibration).

collect_rollouts và train_verifier đều điền trực tiếp GPU="RTX-PRO-6000:1", cpu=4, memory=32768, timeout=3600*24, max_containers=1, retries=0. Collection nhận manifest assets/backbone/calibration, `--policy native|rho` và shard index/count; trainer nhận `--rollout-manifests` dạng một path hoặc danh sách path phân cách dấu phẩy, `--policy` và checkpoint resume tùy chọn. Hai task reload Volume trước khi đọc và commit khi kết thúc/thất bại; trainer commit sau mỗi checkpoint epoch. Runner local là `python collect_rollouts.py` và `python train_verifier.py`, hoặc chạy trực tiếp Python wrapper Modal. Xem [README section 5](../README.md#5-verifier-rollouts-monte-carlo-and-neural-training). Native 128-step trajectory label và rho label có target riêng; không đổi backbone khi train verifier.

Đường gọi của task chuẩn bị dữ liệu: main → run_command.spawn → call.get → src.qm9.dataset.prepare_qm9_assets. Cùng một call được chờ qua cơ chế polling/retry của SDK; không tạo lại job khi chờ kết quả. run_command reload Volume trước khi đọc và commit sau khi ghi, kể cả manifest thất bại. Thuật toán và training nằm trực tiếp trong package src, được khai báo bằng src/__init__.py. Các entrypoint dùng import src.qm9, src.cars, src.sampling và các subpackage tương ứng; pyproject.toml và image Modal giữ cùng namespace này. Không tạo modal/__init__.py để tránh thư mục điều phối che package SDK modal.

Các task chia sẻ một Modal Volume mount tại /root/workspace. Image đặt source/config/metadata tại project root của container (/root), tách khỏi Volume để mount không che source; đường dẫn artifact trong config luôn tương đối từ project root, với tiền tố workspace/. Local dùng cùng cấu trúc workspace/ trong repo:

```text
/root/workspace/
├── data/          # QM9, calibration, rollouts và panel trạng thái
├── model_data/    # Backbone, tokenizer và neural verifier
├── cache/tries/   # Snapshot trie và metadata tương thích
└── results/       # Raw artifact, tests, bảng và đồ thị
```

Artifact được ghi theo ID và manifest; task ghi commit dữ liệu, task đọc reload khi cần thấy cập nhật. Mỗi worker ghi vào đường dẫn run/attempt riêng. Cách lưu này theo [Modal Volumes](https://modal.com/docs/guide/volumes). Giữ riêng chi phí khởi động/image và thời gian tính toán thuật toán trong báo cáo; mọi rollout lồng của MC/CARS vẫn phải được tính vào ngân sách backbone.

resource/ chỉ dùng để đọc tham khảo khi phát triển. Không import, mount, upload hoặc cài code trực tiếp từ resource trong image Modal. Phần code cần dùng lại được sao chép hợp lệ vào src, ghi nguồn/license và chỉnh import cho độc lập; phần mới được viết trong package này. Image cho task verify chứa tests và fixtures của dự án để kiểm chứng ràng buộc đó.

Step 9 dùng lại `collect_rollouts.py`/`run_collect_rollouts.sh` và `train_verifier.py`/`run_train_verifier.sh` với `--policy native-conditional`. Collection nhận thêm `--config configs/native_conditional_rollouts.yaml`; điền count sau profiling trước khi chạy. Native root 32-step phải trùng reference calibration. Mask được chọn độc lập trước khi xem terminal, không phải snapshot diffusion; loader/trainer giữ target/checkpoint riêng với `native` và `rho`. Resource Modal tiếp tục điền trực tiếp như trên, không thêm tqdm vào wrapper. MC/CARS native hiện là thư viện; runner generate/check_values thuộc Step 10. Code/tests Step 9 chưa chạy; xem [README section 9](../README.md#9-native-conditional-value-and-matched-controls).

Thứ tự phụ thuộc chính: prepare_assets → train_base hoặc checkpoint có provenance → calibrate → collect_rollouts → train_verifier. prepare_value_panel dùng backbone và các run nguồn; check_values dùng panel đó. generate chỉ cần neural checkpoint khi cấu hình chọn neural verifier; CARS/MC không phụ thuộc training verifier. verify kiểm tra trước các lượt chạy lớn; benchmark gọi các run đã chốt, rồi evaluate tổng hợp kết quả.

Step 10 đã thêm `generate.py`/`run_generate.sh`, `prepare_value_panel.py`/`run_prepare_value_panel.sh` và `check_values.py`/`run_check_values.sh`. Runner local tương ứng là `python generate.py`, `python prepare_value_panel.py`, `python check_values.py`; lệnh đầy đủ ở [README section 10](../README.md#10-generation-and-fixed-value-panels). Cả ba giữ template App/Image/Volume và tag CUDA Nam yêu cầu, tài nguyên GPU điền trực tiếp RTX-PRO-6000:1, cpu=4, memory=32768, timeout=24h, concurrency=1, retries=0. Check_values có thêm hàm toy với gpu=None, cpu=2, memory=8192, timeout=6h; cả hai hàm dùng cùng image và Volume. Wrapper reload trước khi đọc, commit khi hoàn tất/thất bại, chỉ in kết quả ngắn. Panel/MC/CARS count chưa profile vẫn để null. Code/tests/Modal của Step 10 chưa chạy.

Step 11 thêm `evaluate.py`/`run_evaluate.sh`, tương đương `python evaluate.py` ở repo root. Hàm CPU dùng gpu=None, cpu=2, memory=8192, timeout=6h, concurrency=1, retries=0, cùng CUDA image tag và Volume. Không tải backbone, không sinh mẫu; chỉ đọc artifact/tokenizer/calibration/TRAIN identities, ghi summary attempt mới và commit cả báo cáo incomplete. Chọn `--manifests` hoặc `--run-ids` (mọi attempt, không chọn latest). `--download` nhận tên report artifacts phân cách dấu phẩy; `--download-dir` mặc định `workspace/downloads/evaluation`, kiểm tra hash và không ghi đè local. Wrapper không có tqdm; evaluator bên trong có progress. Incomplete/duplicate-seed selection lưu audit rồi trả lỗi. Xem lệnh ở [README section 11](../README.md#11-evaluate-saved-artifacts). Code/tests/task Step 11 chưa chạy.

Step 12 thêm `verify.py`/`run_verify.sh`, tương đương `python verify.py`. Mặc định CPU: gpu=None, cpu=2, memory=8192, timeout=6h; config/CLI bật `gpu_smoke` mới dùng RTX-PRO-6000:1, cpu=4, memory=32768. Image dùng cùng tag CUDA và allowlist src/configs/tests/modal cùng các runner/file project cần thiết; không chứa resource. Hàm ghi commit cả log thất bại, hàm CPU `read_committed` reload rồi kiểm tra hash log/status ở task khác. Wrapper không có tqdm; pytest processing có progress và giữ mã thoát. Xem [README section 12](../README.md#12-verify-the-offline-pipeline). Code/tests/Modal Step 12 chưa chạy, không có gate nào được đánh dấu passed.

Step 13 adds `benchmark.py`/`run_benchmark.sh`, equivalent to `python benchmark.py`. The CPU coordinator uses gpu=None, cpu=2, memory=8192; the GPU worker uses RTX-PRO-6000:1, cpu=4, memory=32768. Both use timeout=24h, max_containers=1, retries=0, the required CUDA image and the shared workspace Volume. Calls are sequential; each worker checks the locked source/config/input hashes and backend before executing the existing generation/value runner.

The coordinator commits its plan and dispatch record before a worker starts. Workers commit completed or failed artifacts; the coordinator reloads afterward and saves completion manifests, selections and a retry cost ledger. Resume names a previous coordinator manifest and reruns only unfinished units with new attempt IDs. Keep the same backend and config; do not launch concurrent coordinators for one benchmark ID. No tqdm is added to the wrapper.

Commands, prerequisites and pilot/final settings are in [README section 13](../README.md#13-run-the-controlled-benchmark). The config remains unlocked and no Step 13 code, tests, pilot or Modal job has been run. `modal/verify.py` includes the new root runner in its image allowlist so future full verification covers the same source files.

## 14. Shared entrypoint and final workflow

The root `run.py` now selects the existing local task runner. Each of its 11 task names has the Python/shell pair listed above; selecting a task locally uses the current Miniconda environment. The Modal shell scripts continue to call their own `main` entrypoint and forward flags. Task help and importing the local dispatcher create no cloud resources.

Choose local or Modal execution for the same finite toy check:

```bash
python run.py --task check_values --mode toy --value-law native-conditional
bash modal/run_check_values.sh --mode toy --value-law native-conditional
```

Resources below are the constants currently written in the task files. Memory is in MiB; all remote functions use `max_containers=1` and `retries=0`.

| Task / function | GPU | CPUs | Memory | Timeout |
| --- | --- | --- | --- | --- |
| `prepare_assets` | None | 2 | 8192 | 1 hour |
| `train_base` | RTX-PRO-6000:1 | 4 | 32768 | 24 hours |
| `calibrate` | RTX-PRO-6000:1 | 4 | 32768 | 6 hours |
| `collect_rollouts` | RTX-PRO-6000:1 | 4 | 32768 | 24 hours |
| `train_verifier` | RTX-PRO-6000:1 | 4 | 32768 | 24 hours |
| `generate` | RTX-PRO-6000:1 | 4 | 32768 | 24 hours |
| `prepare_value_panel` | RTX-PRO-6000:1 | 4 | 32768 | 24 hours |
| `check_values`, QM9 | RTX-PRO-6000:1 | 4 | 32768 | 24 hours |
| `check_values`, toy | None | 2 | 8192 | 6 hours |
| `verify`, default / readback | None | 2 | 8192 | 6 hours / 10 minutes |
| `verify`, explicit GPU smoke | RTX-PRO-6000:1 | 4 | 32768 | 6 hours |
| `benchmark`, coordinator / worker | None / RTX-PRO-6000:1 | 2 / 4 | 8192 / 32768 | 24 hours each |
| `evaluate` | None | 2 | 8192 | 6 hours |

Every image uses the requested CUDA tag and Python 3.11, with dependencies/files declared directly in its wrapper. The verify/benchmark allowlists now include root `run.py` so their source snapshots cover the same files. Runtime source stays in `src/`; no wrapper imports another Modal entrypoint. Progress remains in task processing.

Prepare assets, finish or explicitly select the backbone checkpoint, then calibrate before generating verifier labels or molecules. `collect_rollouts` and `train_verifier` accept `native`, `rho` and `native-conditional`; the last policy uses `configs/native_conditional_rollouts.yaml` during collection. Individual CARS/MC runs skip neural training; the full comparison needs the upstream-label and matched native-conditional learned controls. Freeze the panel, profile counts/caps, verify current source, then run pilot/final benchmark and evaluate explicit final attempts.

All task artifacts use relative `workspace/` paths on Volume `nam-cars-vgb-qm9-artifacts`, mounted at `/root/workspace`. Writers commit and later readers reload; local workspace data is not uploaded by spelling the same path. Model/data inputs must exist in the selected store. Evaluation downloads selected report artifacts with hash checks; it does not automatically download training checkpoints or raw datasets.

`modal/run_calibrate.sh` currently passes `-d` to the Modal CLI, so that shell returns after submitting a detached job. Its exit code describes submission. Follow the remote job and require the completed calibration manifest before collection/generation; the other task scripts use the attached CLI form shown in their files.

The Step 13 calibration reader/artifact-name mismatch is recorded in [README section 14](../README.md#14-use-the-shared-entrypoint-and-reproduce-a-run) and the implementation checklist. Step 14 code/help/tests/Modal have not been run; no new checkpoint, benchmark result or passed gate is claimed. Earlier paragraphs retain the history of their implementation steps.
