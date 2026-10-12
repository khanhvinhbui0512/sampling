# Nguồn tham khảo và thông báo bên thứ ba

## 1. Phạm vi sử dụng

Các tệp Python của project được viết mới theo kế hoạch và API của thư viện. Bước 4 sao chép prompt/grammar CARS vào `assets/benchmark/`, trích các ví dụ và ghi lại SMARTS gốc; chi tiết ở mục 5. Chưa sao chép mã thuật toán hoặc checkpoint của bên thứ ba vào `src/`.

`resource/` lưu bản tham khảo do người dùng cung cấp, được loại khỏi Git và package runtime. Thư viện trong [pyproject.toml](pyproject.toml) được cài bằng `pip`; project không sao chép mã nguồn của các thư viện đó vào `src/`.

## 2. Bài báo

| Nguồn | Tác giả | Phiên bản và phần tham khảo |
|---|---|---|
| [Constrained Adaptive Rejection Sampling](https://arxiv.org/abs/2510.01902v2) | Paweł Parys, Sairam Vaidya, Taylor Berg-Kirkpatrick, Loris D’Antoni | `2510.01902v2`; Phụ lục B/C cho môi trường và siêu tham số, Phụ lục H cho Molecular Generation (SMILES). Bản được cung cấp: `resource/2510.01902v2.pdf`. |
| [VGB for Masked Diffusion Model: Efficient Test-time Scaling for Reward Satisfaction and Sample Editing](https://arxiv.org/abs/2606.28301v1) | Kijung Jeon, Thuy-Duong Vuong, Molei Tao | `2606.28301v1`; mục 5 và Phụ lục F cho thí nghiệm, mô hình QM9 và process verifier. Bản được cung cấp: `resource/2606.28301v1.pdf`. |

Giấy phép MIT của repository MDM-VGB được ghi ở mục 3 cho mã nguồn. Giấy phép phân phối lại hai PDF chưa được xác minh từ các bản được cung cấp; project dẫn nguồn bài báo và dùng chúng làm tài liệu tham khảo.

## 3. Repository MDM-VGB

- Nguồn chính thức: [KraitGit/MDM-VGB](https://github.com/KraitGit/MDM-VGB).
- Bản được cung cấp: `resource/MDM-VGB/`.
- Commit xác minh từ Git của bản local: `c0e89c7f0d4c83097b2d8ebabd1a325955e1f402`; working tree sạch tại thời điểm ghi nhận.
- Giấy phép: MIT; dòng bản quyền trong `resource/MDM-VGB/LICENSE` là `Copyright (c) 2026 Kijung Jeon`.
- Phần tham khảo cho các bước sau: `resource/MDM-VGB/src/algorithms/`, `resource/MDM-VGB/src/tasks/qm9/` và các cấu hình trong `resource/MDM-VGB/configs/qm9/`. Các tệp mới của bước 1.4–1.5 không dùng registry đa tác vụ của repository này.

Bản giấy phép MIT được giữ nguyên dưới đây để thông tin nguồn có thể đọc độc lập với `resource/`:

```text
MIT License

Copyright (c) 2026 Kijung Jeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 4. Snapshot CARS-code được cung cấp

- Đường dẫn nguồn: `resource/CARS-code/`; metadata đặt tên package là `anon_method`, phiên bản `0.1.0`.
- Giấy phép được khai báo: MIT trong `resource/CARS-code/pyproject.toml`.
- Bản này không có tệp LICENSE, thư mục `.git` hoặc URL repository trong metadata đã đọc. Commit và dòng bản quyền của snapshot chưa xác định; checksum metadata được ghi ở mục 6.
- Phần tham khảo: `resource/CARS-code/src/anon_method/samplers/rejection.py`, các tiện ích trie/grammar và cách dùng mô hình Hugging Face. Project viết riêng các package initializer theo kế hoạch, với import nhẹ.

Thông tin MIT ở mục này chỉ là khai báo của snapshot `CARS-code`; giấy phép của benchmark trong mục 5 được ghi riêng theo nguồn của benchmark.

## 5. Benchmark Molecular Generation (SMILES) của CARS

- Nguồn: [pparys/cars — datasets/smiles](https://github.com/pparys/cars/tree/f46607f983d89e4ca4f40527cd231ab04adc6ed7/datasets/smiles).
- Commit được chọn trong [implementation.md](implementation.md): `f46607f983d89e4ca4f40527cd231ab04adc6ed7`.
- Bản nguồn tham khảo đã đọc không có tệp LICENSE hoặc khai báo giấy phép trong README; giấy phép benchmark chưa xác định từ bản đó.
- Asset đã được sao chép/trích ở giai đoạn 4; mọi checksum và dòng nguồn được ghi trong [source.json](assets/benchmark/source.json).

| Nhóm | Tệp nguồn trong `datasets/smiles/` | Asset đã tạo |
|---|---|---|
| Acrylates | `acrylates.txt`, `acrylates.lark` | `assets/benchmark/acrylates.prompt.txt`, `assets/benchmark/acrylates.examples.smi`, `assets/benchmark/acrylates.lark` |
| Chain extenders | `chain_extenders.txt`, `chain_extenders.lark` | `assets/benchmark/chain_extenders.prompt.txt`, `assets/benchmark/chain_extenders.examples.smi`, `assets/benchmark/chain_extenders.lark` |
| Isocyanates | `isocyanates.txt`, `isocyanates.lark` | `assets/benchmark/isocyanates.prompt.txt`, `assets/benchmark/isocyanates.examples.smi`, `assets/benchmark/isocyanates.lark` |

Prompt và grammar giữ nguyên byte của nguồn tại commit trên. Các tệp `.examples.smi` trích phần sau `Molecule: ` trên những dòng có ví dụ, giữ thứ tự và thêm LF cuối dòng: acrylates dòng 6–37 (32 ví dụ), chain extenders và isocyanates dòng 6–16 (11 ví dụ mỗi nhóm). Không sửa hoặc loại ví dụ theo kết quả checker.

[smarts.json](assets/benchmark/smarts.json) ghi pattern và phép OR của hàm `eval_membership` trong [experiments/smiles/evaluate_molecules.py](https://github.com/pparys/cars/blob/f46607f983d89e4ca4f40527cd231ab04adc6ed7/experiments/smiles/evaluate_molecules.py). SHA-256 của tệp nguồn: `19084bd68ac0b689bc534615a7291bbf75a060ccc39294b8be7cab0611e41f2d`. Chỉ ghi lại pattern và quy tắc matching, không sao chép mã Python. Các ghi chú về matcher chain extender rộng và nhãn chồng lấp là bổ sung của project.

`source.json` được viết mới để lưu provenance. `src/data.py`, `src/tokenizer.py`, `src/grammar.py`, `src/reward.py` và T01 được viết mới; định danh canonical, split/augmentation và reward kết hợp là thiết lập mới. Grammar-only replication giữ tiêu chí grammar của CARS. Project không thay đổi grammar gốc để bắt buộc motif hay ghép cặp ring digits.

## 6. Nhận dạng các bản tham khảo được cung cấp

SHA-256 của các tệp đã đọc:

| Tệp | SHA-256 |
|---|---|
| `resource/2510.01902v2.pdf` | `4d91751338b7b2a283017bee1f1e0ba32664da841e48d9af43fd3587ad18a3e2` |
| `resource/2606.28301v1.pdf` | `0624bc09dedcb6ceb4e51b723fb196d57ac56d017fe344ad17e4249e035c4a3d` |
| `resource/CARS-code/pyproject.toml` | `6d2dfdd1edd2a4ffa4f0e5eaf3f7c9fee3f5a764278bbe728b22f30b136fbc60` |
| `resource/MDM-VGB/pyproject.toml` | `0ee8b8e5805f581133e312b1f107047c8135924c1b8bdb7a6e9b3c1bdb07f458` |
| `resource/MDM-VGB/LICENSE` | `33d779be2daea20a7c0d9e245fe6b0f63ee0165c7c31b5cad0bd1dbade584c50` |

Checksum nhận dạng đúng các tệp trong bảng, không đại diện cho toàn bộ repository hoặc thay thế commit của snapshot.

## 7. Ghi nhận khi đưa nội dung bên thứ ba vào project

Mỗi lần sao chép hoặc chỉnh sửa mã/asset ở các bước sau sẽ bổ sung đường dẫn đích, nguồn và commit, giấy phép cùng thông báo bản quyền, và mô tả phần thay đổi vào tài liệu này. Phần triển khai thuật toán từ mô tả trong paper sẽ ghi rõ là viết mới theo paper. Dữ liệu ChEMBL và checkpoint tải sau sẽ ghi phiên bản, nguồn, giấy phép và checksum trong manifest của artifact tương ứng.

## 8. Nguồn corpus đã cố định trong cấu hình

ChEMBL data is from https://www.ebi.ac.uk/chembl; the version of ChEMBL is chembl_36. [README của bản phát hành](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/README) ghi giấy phép CC Attribution-ShareAlike 3.0 Unported. [configs/data.yaml](configs/data.yaml) chọn `chembl_36_chemreps.txt.gz` (TSV nén, cột `chembl_id` và `canonical_smiles`), với SHA-256 `f9c2fa730ac5000da9b0c0d4033c99bfd8fe2b99950f19df1d842c6e73dd7cc4` từ [checksums.txt](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/checksums.txt). Bước 4 chỉ cố định nguồn; task tải/chuẩn bị corpus thuộc bước 5. Corpus này là dữ liệu bổ sung mới, khác dữ liệu few-shot của CARS và QM9 của MDM-VGB.

## 9. Bước 8 — CARS và nguồn đã đọc

`src/trie.py`, `src/cars.py`, T09 local/Modal, cấu hình và kiểm thử được viết mới từ thuật toán CARS; không sao chép mã Python của tác giả. Đã đọc [CARS v2, mục 3 và Phụ lục C/H](https://arxiv.org/html/2510.01902v2), [MDM-VGB v1, định nghĩa value và terminal anchoring](https://arxiv.org/html/2606.28301v1), cùng các tệp nguồn dưới đây. Các bản local trong `resource/` không có trong workspace ở bước này; nguồn trực tuyến đúng revision được dùng thay thế để đọc. Runtime chỉ dùng mã, asset và artifact của project.

| Tệp nguồn đã đọc | Revision | SHA-256 |
|---|---|---|
| [cars/cars.py](https://github.com/pparys/cars/blob/f46607f983d89e4ca4f40527cd231ab04adc6ed7/cars/cars.py) | `f46607f983d89e4ca4f40527cd231ab04adc6ed7` | `30dd6c8a11253197e46d68ff2ad44dd40a5ff636ca9c7a03f55bba3b0a9cff9e` |
| [cars/logits_process.py](https://github.com/pparys/cars/blob/f46607f983d89e4ca4f40527cd231ab04adc6ed7/cars/logits_process.py) | Cùng revision CARS | `c294bd4f1593b08cfa9fcf0393f1948fc02ee3f951a1e9a9b10f8e50b3f6cf90` |
| [cars/lib.py](https://github.com/pparys/cars/blob/f46607f983d89e4ca4f40527cd231ab04adc6ed7/cars/lib.py) | Cùng revision CARS | `1ce5ed8cd2b1a6b58a8e40dfecbbbccbefbe1ba48a5c89d7a7ccfd13f70b046a` |
| [llguidance_grammar_recognizer.py](https://github.com/pparys/cars/blob/f46607f983d89e4ca4f40527cd231ab04adc6ed7/llguidance_grammar_recognizer.py) | Cùng revision CARS | `2a5b815797cbe5104dab117bf0bdeb3dfce6f8a62023760076c7ed8b3c05f1c9` |
| [src/algorithms/algorithm_utils.py](https://github.com/KraitGit/MDM-VGB/blob/c0e89c7f0d4c83097b2d8ebabd1a325955e1f402/src/algorithms/algorithm_utils.py) | `c0e89c7f0d4c83097b2d8ebabd1a325955e1f402` | `5f7cc7fedfb83a113aa541ff9271d1e098e3b0bb13896af9573b363ed4d849c2` |

Giấy phép repository CARS tại revision trên chưa được xác định; không suy ra giấy phép của nó từ snapshot `CARS-code` ở mục 4. Phần viết mới dùng logsumexp để tính remaining mass, học sau attempt để giữ proposal cố định, và chỉ loại lá khi reward RDKit/SMARTS bằng 0. Root constraint, phản hồi toàn vocabulary và giới hạn AR nhận chuỗi grammar-complete tại cap được đối chiếu với source. Adapter q_z giữ known suffix, reward kết hợp, giới hạn số node, resume và kiểm kê chi phí là phần thích nghi của project. Bước 8 chưa triển khai value provider hoặc kernel MDM-VGB của Bước 9.

## 10. Bước 9 — Value providers và kernel MDM-VGB

Đã đọc [MDM-VGB v1, Định nghĩa 4.2, mục 4.3 và Phụ lục E/F](https://arxiv.org/html/2606.28301v1), cùng thuật toán và mass của [CARS v2](https://arxiv.org/html/2510.01902v2). Vì `resource/` không có trong workspace, nguồn MDM-VGB dưới đây được đọc trực tuyến tại commit `c0e89c7f0d4c83097b2d8ebabd1a325955e1f402`; giấy phép MIT và dòng bản quyền của nguồn được giữ ở mục 3. Mã project được viết riêng theo API hiện có.

| Tệp nguồn đã đọc | SHA-256 |
|---|---|
| [src/algorithms/vgb.py](https://github.com/KraitGit/MDM-VGB/blob/c0e89c7f0d4c83097b2d8ebabd1a325955e1f402/src/algorithms/vgb.py) | `5c3d215719c953e736ff096c9979ffbac73d5ca4bc02e4c02f84f27c7d5b60ed` |
| [src/algorithms/vgb_candidates.py](https://github.com/KraitGit/MDM-VGB/blob/c0e89c7f0d4c83097b2d8ebabd1a325955e1f402/src/algorithms/vgb_candidates.py) | `a613782f667919ddfe4c6e9394ab06e3dda82fecf6836b9943ab808fc9f1a99b` |
| [src/algorithms/vgb_terminal.py](https://github.com/KraitGit/MDM-VGB/blob/c0e89c7f0d4c83097b2d8ebabd1a325955e1f402/src/algorithms/vgb_terminal.py) | `d6ab024b3f72e3dccb8ecfe69c2719f3bc18852a8cae0580f2825af8c14d996e` |
| [src/algorithms/vgb_momentum.py](https://github.com/KraitGit/MDM-VGB/blob/c0e89c7f0d4c83097b2d8ebabd1a325955e1f402/src/algorithms/vgb_momentum.py) | `7fb5d044bf570287af59e2d11f3623f0f4b338a9000510e77e1f231e9a7d1c36` |
| [configs/qm9/qm9_inference.yaml](https://github.com/KraitGit/MDM-VGB/blob/c0e89c7f0d4c83097b2d8ebabd1a325955e1f402/configs/qm9/qm9_inference.yaml) | `20cd4a1f86fc5ce2bc1885d449275279bfaccfe7cd9217c43e17e28179d9846a` |
| [src/tasks/qm9/harness.py](https://github.com/KraitGit/MDM-VGB/blob/c0e89c7f0d4c83097b2d8ebabd1a325955e1f402/src/tasks/qm9/harness.py) | `994f9fdbc61120413ad16fa2b6383e306d96dfb325d189670ef41da5f42fd1f1` |

Kernel dùng hệ số nghịch đảo tổ hợp theo độ sâu, shortlist lấy mẫu có hoàn lại và gộp multiplicity như `build_mdm_candidates`; lambda mặc định 0. Momentum tùy chọn dùng residual directional mass với chi=1. Chế độ `K=all` của project dùng xác suất base thật khi liệt kê, thay vì trọng số đều. Floor chỉ áp dụng state chưa hoàn chỉnh; leaf-zero luôn có trọng số 0 và gate lambda được bỏ tại biên terminal theo paper. Confidence remask đọc logits denoiser trước absorbing copy. Fallback mặc định tắt theo `terminal.force_complete=false` của QM9, có thể bật rõ ràng và được tính phí riêng.

EOS/PAD, reward kết hợp, input observations cố định, cache hữu hạn, kiểm tra provenance, resume giữa bước, nN tách khỏi NFE và fallback forward có tính phí là các thích nghi SMILES. Không kế thừa khẳng định stationary law của đồ thị đầy đủ cho shortlist thực tế. CARS value dùng m cố định và mass trước draw, không dùng accepted-only average; budget ngắt query không xuất estimate một phần. CI CARS dùng Hoeffding–Azuma cho đóng góp thích nghi bị chặn; reference MC độc lập dùng Wilson và báo SE riêng. Learned provider giữ nguyên loại target của rollout huấn luyện, CARS báo V_q; cả hai không được gọi là V* lý tưởng. Bước 9 chưa triển khai đánh giá T13 hoặc metrics/report ở các bước sau.

## 11. Bước 10 — Đánh giá value trực tiếp

Đã đọc lại [MDM-VGB v1, Phụ lục B: Completion values và Verifier training](https://arxiv.org/html/2606.28301v1), [CARS v2, Algorithm 1 và tracking mass](https://arxiv.org/html/2510.01902v2), mã QM9 tại revision/checksum ở mục 10 và các module completion/value/verifier/tokenized shards của project. T13 được viết mới; không sao chép mã nguồn bên thứ ba và không đọc `resource/` khi chạy.

Chọn tối đa 16 phân tử nguồn mỗi nhóm, mask theo fixed-width, partition ID state cho tuning/final, loại trùng với identities và snapshots của verifier, 512 reference completions, batch 32, ba repetition, và hạn mức NFE/thời gian trong `configs/value_eval.yaml` là protocol đánh giá mới. CARS vẫn dùng m cố định; cache tắt trong các query đo chi phí. Hai provider có cùng hạn mức nhưng báo riêng chi phí thực dùng, verifier calls và reference cost; không suy ra adjusted NFE khi chưa profile FLOPs.

Reference q_z và reverse rollout được đo riêng từ cùng state với seed độc lập, terminal anchoring và Wilson CI. Reverse completion từ partial state bắt đầu lại ở time 1; việc trùng training-policy hash không chứng minh conditional target của snapshot regression trùng reference này. MSE/MAE/calibration dùng MC hữu hạn, ghi độ bất định và số query thiếu, không gọi reference là V*. Shard và state pool được ghim checksum; resume giữ query đã commit và chạy lại query dở dang theo seed cố định, giữ riêng chi phí ngắt đã ghi nhận. Bước 10 không triển khai metrics phân tử, retrosynthesis hay báo cáo tổng hợp của Bước 11.

## 12. Bước 11 — Metrics, RetroStar và báo cáo

Đã đọc [CARS v2, Phụ lục H.4](https://arxiv.org/html/2510.01902v2#A8.SS4), [MDM-VGB v1, Phụ lục F về raw/FLOP-adjusted NFE](https://arxiv.org/html/2606.28301v1), và các nguồn được ghim bên dưới. Module metrics/reporting của project được viết riêng, không import hoặc đọc `resource/` khi chạy.

| Tệp tham chiếu | Revision | SHA-256 |
|---|---|---|
| [CARS evaluate_molecules.py](https://github.com/pparys/cars/blob/f46607f983d89e4ca4f40527cd231ab04adc6ed7/experiments/smiles/evaluate_molecules.py) | `f46607f983d89e4ca4f40527cd231ab04adc6ed7` | `19084bd68ac0b689bc534615a7291bbf75a060ccc39294b8be7cab0611e41f2d` |
| [RetroStar backward_model.py](https://github.com/AustinT/syntheseus-retro-star-benchmark/blob/2f4e2fa20a5c83c667d7b4de5bce3a35f1420520/syntheseus_retro_star_benchmark/backward_model.py) | `2f4e2fa20a5c83c667d7b4de5bce3a35f1420520` | `a9018ac96a5cae306a22b25cb584826b431fa967a935bb31d1f04ef34f64f91a` |
| [RetroStar mlp_inference.py](https://github.com/AustinT/syntheseus-retro-star-benchmark/blob/2f4e2fa20a5c83c667d7b4de5bce3a35f1420520/syntheseus_retro_star_benchmark/original_code/mlp_inference.py) | Cùng revision RetroStar | `9db7cd1c95f5af56999ef8c6214fc3ed0a05522fa077bc822b15c95afaa01cae` |
| [RetroStar mlp_policies.py](https://github.com/AustinT/syntheseus-retro-star-benchmark/blob/2f4e2fa20a5c83c667d7b4de5bce3a35f1420520/syntheseus_retro_star_benchmark/original_code/mlp_policies.py) | Cùng revision RetroStar | `82fa17967a87f5f26713ac5d51ad7e2c77884a7837ce01954c5d766c52cea511` |

Dependency `syntheseus-retro-star-benchmark` được cài từ đúng revision MIT đã khai báo trong T04; không vendor mã của nó vào project. Adapter truyền rõ model/template paths đã kiểm checksum, expansion top-k 50, không shuffle, gọi đúng top-1 reaction output và đọc probability sau bước áp dụng/merge/normalize reaction templates của upstream. Không thay proxy bằng xác suất softmax của template trước khi áp dụng phản ứng.

Mã CARS gọi threshold 0,4 và nhân tỷ lệ trên valid molecules với validity, tương đương số output có top-1 probability > 0,4 chia cho toàn bộ output. Project giữ mẫu số này và báo thêm mẫu số valid-only. Paper dùng Morgan radius 2/2.048 bits, mã gốc dùng radius 3; project dùng radius 2 làm chính và radius 3 làm sensitivity. Novelty dùng canonical molecular identity thay fingerprint equality. Chọn mẫu số root attempts, tách shortlist candidates/selected set, seed bootstrap, paired caps, checksum/audit và báo unavailable là bổ sung của project. Không dùng QED/Lipinski làm metric chính, không suy ra route tổng hợp nhiều bước hay V*.

## 13. Bước 12 — Điều phối comparison

Đã đọc lại [CARS v2, Phụ lục H.4](https://arxiv.org/html/2510.01902v2#A8.SS4), [MDM-VGB v1, Phụ lục F](https://arxiv.org/html/2606.28301v1), cùng mã generation, tracking, evaluation, reporting và Modal của project. T15 được viết mới, không sao chép bộ điều phối bên thứ ba và không đọc `resource/` khi chạy.

Matrix ba phương pháp và baseline, allowances NFE/thời gian, Cartesian N/m validation sweep, tái sử dụng job trùng, selection manifest, checksum và cơ chế resume là thiết kế của project. NFE dùng tổng forward thật, gồm inner hybrid, giữ riêng verifier calls; không coi N là NFE hay ngoại suy run chưa đạt mục tiêu. Validation generation dùng seed cộng 2³¹, final dùng seed gốc trên cùng prompt; đây không phải split phân tử held-out. Cấu hình/checkpoint chỉ được khóa sau lựa chọn tường minh, T13 validation/test vẫn dùng split state riêng. Suite replication dùng AR/reward riêng. Coordinator CPU dispatch worker riêng và commit trước đánh giá; không đưa ra khẳng định V* hay hiệu năng paper từ các allowance mới.
