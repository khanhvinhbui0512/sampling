# Tổ chức các task chạy trên Modal

Modal là môi trường thực thi của dự án. Tài liệu này chốt cách tổ chức ở STEP 4; các file Python và shell dưới đây sẽ được viết ở bước IMPLEMENT. Hiện thư mục này chỉ chứa README.md.

Mỗi task có một file Python và một script shell tương ứng. File Python cung cấp entrypoint main để nhận tham số và gọi package cars_vgb; script gọi modal run modal/<task>.py::main. Cú pháp file::entrypoint được mô tả trong [Modal CLI](https://modal.com/docs/cli/latest/run).

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

Mọi script sẽ xác định project root, chuyển tiếp nguyên vẹn tham số người dùng và trả đúng exit code của Modal. Mỗi entrypoint sẽ kiểm tra config, artifact đầu vào và ghi cấu hình đã resolve cùng run ID. benchmark chỉ điều phối; GPU được cấp cho worker cần model. prepare_assets, evaluate và kiểm chứng toy dùng CPU; training và generation dùng GPU, các task hỗn hợp chọn tài nguyên theo chế độ chạy. Loại GPU, timeout và concurrency sẽ nằm trong configs/modal.yaml, chưa chốt trước profiling.

Cấu hình image, dependency và Volume dùng chung sẽ nằm tại src/cars_vgb/execution/modal_runtime.py. Thuật toán và training nằm trong src/cars_vgb, không được nhân bản trong các entrypoint. Không tạo modal/__init__.py để tránh thư mục điều phối che package SDK modal.

Các task chia sẻ một Modal Volume mount tại /workspace:

```text
/workspace/
├── data/          # QM9, calibration, rollouts và panel trạng thái
├── model_data/    # Backbone, tokenizer và neural verifier
├── cache/tries/   # Snapshot trie và metadata tương thích
└── results/       # Raw artifact, tests, bảng và đồ thị
```

Artifact được ghi theo ID và manifest; task ghi commit dữ liệu, task đọc reload khi cần thấy cập nhật. Mỗi worker ghi vào đường dẫn run/attempt riêng. Cách lưu này theo [Modal Volumes](https://modal.com/docs/guide/volumes). Giữ riêng chi phí khởi động/image và thời gian tính toán thuật toán trong báo cáo; mọi rollout lồng của MC/CARS vẫn phải được tính vào ngân sách backbone.

resource/ chỉ dùng để đọc tham khảo khi phát triển. Không import, mount, upload hoặc cài code trực tiếp từ resource trong image Modal. Phần code cần dùng lại được sao chép hợp lệ vào src/cars_vgb, ghi nguồn/license và chỉnh import cho độc lập; phần mới được viết trong package này. Image cho task verify chứa tests và fixtures của dự án để kiểm chứng ràng buộc đó.

Thứ tự phụ thuộc chính: prepare_assets → train_base hoặc checkpoint có provenance → calibrate → collect_rollouts → train_verifier. prepare_value_panel dùng backbone và các run nguồn; check_values dùng panel đó. generate chỉ cần neural checkpoint khi cấu hình chọn neural verifier; CARS/MC không phụ thuộc training verifier. verify kiểm tra trước các lượt chạy lớn; benchmark gọi các run đã chốt, rồi evaluate tổng hợp kết quả.
