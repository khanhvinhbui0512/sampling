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
