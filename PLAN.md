# Kế hoạch: CARS, MDM-VGB và phương pháp kết hợp trên Molecular Generation (SMILES)

## 1. Mục tiêu và phạm vi

So sánh ba phương pháp trên **cùng benchmark Molecular Generation (SMILES) của bài CARS**: acrylates, chain extenders và isocyanates. Ba phương pháp chính là:

1. **CARS:** lấy mẫu có ràng buộc bằng trie loại các nhánh vi phạm.
2. **MDM-VGB:** sinh và sửa SMILES bằng mô hình khuếch tán che token, dùng process verifier được huấn luyện để ước lượng giá trị trạng thái.
3. **CARS + MDM-VGB:** giữ cơ chế chuyển trạng thái của MDM-VGB, thay giá trị từ verifier bằng ước lượng dựa trên CARS.

Mục tiêu nghiên cứu riêng của phương pháp kết hợp là đánh giá liệu CARS có thể ước lượng hữu ích giá trị $V^*(z)$ của một SMILES bị che một phần $z$, xét theo **cùng mô hình nền và cùng ràng buộc** với MDM-VGB. QM9 trong mã MDM-VGB chỉ là nguồn tham khảo về kiến trúc và quy trình huấn luyện; QM9 không phải benchmark của project. QED và Lipinski không phải chỉ số chính vì Phụ lục H của bài CARS chủ động không dùng chúng cho các tiền chất polymer và hóa chất chuyên dụng.

## 2. Cố định benchmark trước khi huấn luyện

- Tạo một đặc tả bất biến cho ba nhóm: toàn bộ 32 ví dụ acrylate, 11 ví dụ chain extender và 11 ví dụ isocyanate; prompt gốc; ngữ pháp SMILES; biểu thức SMARTS để kiểm tra nhóm; phiên bản công cụ và quy tắc chuẩn hóa SMILES.
- Lấy prompt và ngữ pháp đầy đủ từ kho thí nghiệm gốc `https://github.com/pparys/cars`, thư mục `datasets/smiles/` (đã kiểm tra tại commit `f46607f983d89e4ca4f40527cd231ab04adc6ed7`); thư mục `resource/CARS-code/` được cung cấp chỉ là thư viện sampler tổng quát. Cố định commit, lưu checksum và kiểm tra số ví dụ 32/11/11. Hình 7 trong PDF chỉ chứa bản rút gọn.
- Kiểm tra ngữ pháp ở **mức token của tokenizer được dùng**: nhận các ví dụ hợp lệ, loại các chuỗi vi phạm được biết, có thể kết thúc đúng cách, và không loại nhầm chuỗi mà hàm thưởng coi là thành công.
- Tách ba khái niệm: *được ngữ pháp chấp nhận*, *RDKit phân tích được*, và *thuộc nhóm hóa học theo SMARTS*. Một chuỗi thỏa ngữ pháp vẫn có thể không hợp lệ về hóa học; các chỉ số này không được gộp ngầm.
- Ngữ pháp trong kho thí nghiệm gốc chỉ đưa motif nhóm hóa học vào một lựa chọn `GROUP_SYMBOL`; nó không bắt buộc motif xuất hiện. Vì vậy reward “được ngữ pháp chấp nhận” không đủ để giải mục tiêu sinh đúng nhóm. Ghi riêng sai khác giữa mô tả của bài báo, ngữ pháp thực tế và SMARTS của bộ đánh giá.
- Bộ đánh giá gốc dùng SMARTS acrylate `C=CC(=O)O*`, isocyanate `[*]N=C=O`, và chain extender khớp một trong `CO`, `OC`, `N`. Giữ các pattern này để so với benchmark, đồng thời nêu rõ bộ kiểm tra chain extender không bảo đảm có hai nhóm chức phản ứng như mô tả hóa học trong bài. Nếu kiểm tra hai nhóm chức nghiêm ngặt hơn thì coi đó là phân tích phụ.
- Chuẩn hóa bằng RDKit để phát hiện cùng một phân tử được viết bằng nhiều SMILES khác nhau. Loại các phân tử trùng với ví dụ trong prompt khi tính tính mới và số lượng phân tử duy nhất. Ghi cả SMILES gốc lẫn SMILES chuẩn hóa.

## 3. Định nghĩa bài toán chung

Trong so sánh chính, một ứng viên chỉ thành công khi vừa thỏa ngữ pháp benchmark $G_c$, vừa được RDKit phân tích hợp lệ, vừa thuộc nhóm $c$ theo SMARTS đã chốt. Dùng cùng hàm thưởng nhị phân và cùng bộ kiểm tra cho cả ba phương pháp:

$$
\tau_c(y)=\mathbf{1}\{y\in G_c,\ \mathrm{RDKitValid}(y),\ \mathrm{SMARTS}_c(y)\}.
$$

Với trạng thái bị che $z$, giá trị lý tưởng cần xấp xỉ là xác suất một lần hoàn thiện theo chính sách tham chiếu tạo được phân tử thành công:

$$
V_c^*(z)=\mathbb{E}_{Y\sim\pi_{\mathrm{ref}}(\cdot\mid z,c)}[\tau_c(Y)].
$$

Đây là **thay đổi có chủ đích so với bài CARS**: ngữ pháp trong kho tác giả chỉ cho phép motif nhóm hóa học chứ không buộc motif xuất hiện, nên chỉ thưởng grammar membership sẽ không bảo đảm sinh đúng nhóm. Việc loại một prefix chỉ dựa trên chứng minh nó không thể thỏa ngữ pháp vẫn an toàn vì mọi đầu ra có reward 1 đều phải thỏa ngữ pháp; RDKit và SMARTS được kiểm tra thêm tại terminal. Khóa cả mẫu số của reward và quy tắc xử lý lỗi RDKit trước khi chạy.

Chạy một nhánh **tái lập theo bài CARS** riêng, dùng reward chỉ dựa vào grammar, thu 100 mẫu duy nhất được grammar chấp nhận rồi đo RDKit validity và membership. Không gộp các số liệu này với thí nghiệm chính. Vì tập thành công của thí nghiệm chính đương nhiên có validity và membership 100%, các tỷ lệ validity/membership để so sánh phải tính trên **toàn bộ lần sinh hoàn chỉnh**, đồng thời báo time/attempts để đạt 100 phân tử thành công duy nhất.

## 4. Mô hình nền và tính công bằng

Thí nghiệm so sánh chính dùng **cùng một masked diffusion model (MDLM)**, tokenizer, dữ liệu huấn luyện, điều kiện nhóm hóa học và bộ kiểm tra terminal cho cả ba phương pháp. Để áp dụng CARS cho MDLM, định nghĩa một chính sách hoàn thiện tuần tự: từ $z$, mở các vị trí còn che theo thứ tự trái sang phải, lấy xác suất token từ MDLM tại trạng thái hiện tại, đồng thời đưa các token đã hiện vào bộ nhận dạng ngữ pháp. Chính sách này tạo ra một phân phối hoàn thiện xác định $q_z$, có thể tính xác suất nhánh cho trie CARS. CARS độc lập và CARS trong phương pháp kết hợp dùng đúng chính sách đó.

Mở rộng khung QM9 của MDM-VGB cho SMILES dài hơn và ba nhóm hóa học: tokenizer có token kết thúc và padding rõ ràng, độ dài tối đa được chốt trước khi đánh giá, giải mã nhất quán và class conditioning. Giữ nguyên cơ chế unmask/remask, shortlisting và terminal anchoring của MDM-VGB; chỉ đổi phần đặc thù của tác vụ. Không huấn luyện mô hình chỉ trên 54 ví dụ của prompt. Chọn một bản phát hành ChEMBL công khai và cố định, đề xuất ChEMBL 36, làm corpus ban đầu; tải nguồn chính thức, lọc và chuẩn hóa bằng RDKit, gán nhãn nhóm bằng SMARTS đã chốt, tách train/validation/test theo phân tử chuẩn hóa **trước** khi tạo các SMILES ngẫu nhiên, và loại trùng với ví dụ benchmark bằng định danh phân tử. Đo số lượng và độ phủ của mỗi nhóm trước khi quyết định cách lấy mẫu/điều kiện hóa. Ghi nguồn, giấy phép, quy tắc lọc, độ phủ độ dài và tỷ lệ mỗi nhóm. Đây là dữ liệu huấn luyện mới; bài CARS chỉ dùng ví dụ trong prompt, còn bài MDM-VGB huấn luyện trên QM9.

Trong so sánh chính không giả định đã có pretrained SMILES MDLM phù hợp: huấn luyện một MDLM chung từ đầu rồi dùng **cùng checkpoint** cho ba phương pháp. Lấy cấu hình QM9 làm điểm khởi đầu (12 lớp, hidden 768, 12 head, dropout 0,1, 25.000 bước, batch toàn cục 256, learning rate 3e-4, warmup 1.000), sau đó chốt độ dài tối đa và điều kiện nhóm theo corpus/validation. Đề xuất chia corpus 80/10/10 theo phân tử với seed cố định, dùng độ dài ban đầu $n=256$ token SMILES và báo tỷ lệ bị loại do độ dài; đây không phải cùng đơn vị token với tokenizer Llama. Mô hình NV-GenMol sinh SAFE và các SMILES masked-language encoder không thể thay trực tiếp bộ sinh MDLM cho ngữ pháp SMILES. Nhánh tái lập CARS gốc dùng Llama-3.1-8B-Instruct theo checkpoint nêu trong bài, tách khỏi so sánh chính.

**Giới hạn lý thuyết cần kiểm tra:** xác suất token tại một trạng thái che bất kỳ của MDLM không tự động là conditional chính xác của một phân phối chung duy nhất. Vì vậy ước lượng CARS trước hết nhắm tới giá trị theo chính sách hoàn thiện $q_z$. Chỉ gọi nó là xấp xỉ $V^*$ theo phân phối tham chiếu của bài MDM-VGB nếu kiểm tra tính nhất quán của hai phân phối cho kết quả đủ tốt; nếu không, báo cáo rõ là *giá trị thay thế theo $q_z$*. Không chuyển nguyên tuyên bố “exact” của CARS trên mô hình autoregressive sang MDLM khi điều kiện này chưa được chứng minh.

Để đối chiếu bài CARS gốc, chạy thêm CARS với mô hình autoregressive, prompt few-shot và giới hạn 256 token như bài báo khi checkpoint Llama được cấp quyền truy cập. Đây là nhánh tái lập phụ, **không** trộn số liệu của nó vào so sánh ba phương pháp dùng chung MDLM, vì mô hình nền khác nhau. Cố định temperature 1, top-p 1, top-k 0 và BF16. Mã `run_task.py` trong kho tác giả mặc định 512 token và 2.000 bước, còn Phụ lục C/H nêu 256 token và timeout 1.000 lần sinh; wrapper thí nghiệm phải đặt rõ cấu hình theo bài báo và ghi khác biệt. Script `run_smiles_tasks_all.sh` thiếu tham số model khi gọi `run_task.py`, nên không coi nó là quy trình tái lập chạy nguyên trạng.

Tổ chức mã cho một benchmark: các module dùng chung nằm trực tiếp trong `src/`, các task local nằm trong `src/tasks/`, và mỗi task có một wrapper cùng tên trong `modal/`. `src/__init__.py` khai báo package `src`; dùng import `src.*` và lệnh local `python src/tasks/<task>.py` sau khi cài package editable. Lệnh cloud là `python modal/<task>.py`, với `App`, `Image`, `Volume`, `@app.function` và `app.run()`. Local dùng Miniconda với Python 3.12 và `pip`; image Modal dùng tag `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, Python 3.12 và `Image.pip_install` với dependencies đọc từ `pyproject.toml`. Ghi phiên bản thực tế trong báo cáo môi trường; `requirements.lock` là tùy chọn. Cây thư mục và thứ tự hoàn thiện từng tệp được mô tả trong [implementation.md](implementation.md).

## 5. Triển khai ba phương pháp

### 5.1. CARS trên chính sách hoàn thiện MDLM

- Chuyển các token được mở tuần tự thành đường đi trong trie. Lưu xác suất nhánh và khối xác suất còn lại sau khi loại các tiền tố chắc chắn không thể hoàn thành thành phân tử có reward 1.
- Dùng bộ kiểm tra prefix của ngữ pháp khi nó có thể chứng minh nhánh vi phạm; nếu không chứng minh được, tiếp tục lấy mẫu và kiểm tra ở terminal. Chỉ loại nhánh đã biết chắc có reward bằng 0 để bảo toàn phép tính xác suất.
- Cho phép CARS học từ cả mẫu bị từ chối và các token kế tiếp không hợp lệ tại các prefix đã thăm. Lá hoàn chỉnh không thỏa RDKit hoặc SMARTS được đánh dấu thất bại, không suy diễn rằng mọi nhánh cùng tiền tố đều thất bại. Kiểm tra chuẩn hóa xác suất trie sau mỗi lần cập nhật. Bộ thu thập mẫu bên ngoài sampler xử lý loại trùng, ví dụ prompt và giới hạn nỗ lực chung.

### 5.2. MDM-VGB với verifier học được

- Dùng MDLM đã huấn luyện làm mô hình nền cho unmask/remask. Sinh rollout gốc và lưu các trạng thái che ở nhiều độ sâu cùng reward terminal $\tau_c$.
- Huấn luyện verifier trên các trạng thái đó bằng hồi quy reward, có validation tách theo phân tử và kiểm tra calibration theo nhóm/độ sâu che. Dùng reward thật tại lá hoàn chỉnh (terminal anchoring).
- Bắt đầu từ trạng thái che toàn bộ, chạy MDM-VGB với bộ siêu tham số và quy tắc dừng đã chốt trên validation. Cấu hình QM9 gốc để khởi đầu: $L_f=8$, $L_b=32$, $K=8$, block $B=1$, $\lambda=0$, nhiệt độ 1, chọn forward ngẫu nhiên/backward theo low confidence, momentum $\chi=1$, $N=16$ và tối đa $nN$ bước rồi hoàn thiện chỉ theo hướng forward. Không chép nguyên $n=32$, vocabulary 35 hoặc ngưỡng QED của QM9 sang benchmark mới. Thử $N\in\{1,2,4,8,16\}$ và shortlist/budget value trên validation. Rollout QM9 gốc dùng 128 bước, 5 snapshot mỗi rollout; verifier tham chiếu có 4 lớp, hidden 256, 4 head, dropout 0,1, sigmoid, 200 epoch, batch 2.048, learning rate 1e-4; chọn checkpoint cuối theo validation. Bảo đảm `terminal_accept` khớp chính xác $\tau_c$, vì mã QM9 gốc chỉ kiểm tra RDKit-valid ở hàm này dù reward hướng dẫn dựa trên QED. Giữ một baseline sinh MDLM không guidance để đo mức khó của từng nhóm.

### 5.3. CARS ước lượng value cho MDM-VGB

- Ở mỗi trạng thái hoặc ứng viên mà MDM-VGB cần value, khởi tạo hoặc truy xuất trie CARS tương ứng với $(z,c)$, rồi lấy các completion theo $q_z$. Cache theo trạng thái che đã chuẩn hóa và nhóm; giới hạn bộ nhớ và số completion cho mỗi lần hỏi.
- Nếu $W_t$ chỉ gồm các nhánh chắc chắn thất bại, đặt $p_t=q_z(\text{tránh }W_t)$. Với một completion $Y_t$ lấy từ phân phối CARS **trước khi cập nhật $W_t$**, lượng $p_t\tau_c(Y_t)$ có kỳ vọng bằng $\mathbb{E}_{q_z}[\tau_c(Y)]$, kể cả khi trie được cập nhật giữa các lần lấy mẫu. Trung bình các lượng này để ước lượng value; đồng thời theo dõi $p_t$ như cận trên và báo cáo độ bất định của ước lượng. Phép suy luận này phải được kiểm tra bằng ví dụ hữu hạn trước khi dùng trên SMILES.
- Cài bộ ước lượng qua giao diện `verifier.values(...)` mà `values_for_states` gọi cho trạng thái chưa hoàn chỉnh; giữ xử lý reward thật ở terminal trong `values_for_states`. Giữ thuật toán chuyển trạng thái, mô hình nền, reward terminal và ngân sách tìm kiếm như phương pháp 5.2. Dùng batch và cache để kiểm soát chi phí. Với value rất nhỏ, dùng cùng quy tắc epsilon/terminal anchoring đã chốt cho cả baseline và phương pháp kết hợp.
- Khởi đầu với $m=16$ completion/attempt cho mỗi value query, thử $m\in\{4,8,16,32\}$ trên validation và dùng $m$ cố định trong thí nghiệm chính. Ghi **mọi attempt**, kể cả thất bại; trung bình trên các mẫu đã được CARS chấp nhận riêng sẽ sai mục tiêu. Nếu thử dừng sớm thích ứng thì báo riêng vì tính không chệch của trung bình với $m$ cố định không tự động còn đúng. Không tính các completion này là miễn phí: cộng mọi lần gọi MDLM, kiểm tra ngữ pháp và reward vào chi phí phương pháp kết hợp.

## 6. Quy trình đánh giá thống nhất

- Chạy riêng cho ba nhóm và ít nhất ba seed độc lập. Mỗi phương pháp có cùng tập điều kiện nhóm, ngữ pháp, bộ đánh giá và quy tắc hậu xử lý. Chốt mọi siêu tham số trên validation, sau đó khóa cấu hình trước khi chạy test.
- Trong so sánh chính, thu thập đến 100 phân tử **duy nhất có reward 1**, loại ví dụ prompt, với giới hạn 1.000 lượt thử sinh mỗi nhóm và mỗi seed. Đếm cả lượt CARS bị cắt tại prefix; ghi riêng số đầu ra hoàn chỉnh, lượt khởi động từ root và số lá thăm trong một lần chạy VGB. Nhánh tái lập CARS theo Phụ lục H thu 100 đầu ra duy nhất **được grammar chấp nhận** với cùng cap. Nếu không đạt 100, báo số thực đạt và tỷ lệ timeout; không thay số thiếu bằng ngoại suy mà không ghi rõ.
- Tính RDKit validity và class membership bằng cùng SMARTS trên **toàn bộ ứng viên hoàn chỉnh**; diversity là khoảng cách Tanimoto trung bình từng cặp trên Morgan fingerprint bán kính 2, 2.048 bit. Mã đánh giá gốc của tác giả dùng bán kính 3, khác mô tả bán kính 2 trong Phụ lục H: dùng 2 làm chỉ số chính và báo 3 như phân tích độ nhạy nếu cần. Cố định rõ phiên bản và công thức retrosynthesis score; mã gốc dùng xác suất phản ứng bước đầu của RetroStar trên ngưỡng 0,4, không mặc nhiên đồng nhất với điểm tìm lộ trình nhiều bước. Công bố mẫu số và tập ứng viên áp dụng cho từng chỉ số.
- Báo số lần sinh đến 100 mẫu, tỷ lệ chấp nhận, số phân tử chuẩn hóa duy nhất, độ mới, raw NFE của MDLM, số lần gọi verifier/ước lượng value, FLOPs quy đổi, thời gian chạy và bộ nhớ. Phân biệt “số lần sinh” ở Bảng 15 với “số forward pass” được mô tả trong phần chỉ số của Phụ lục H; lưu cả hai thay vì coi chúng là một.
- So sánh chất lượng tại **ngân sách tính toán tương ứng**, không chỉ tại cùng số đầu ra: dựng đường cong chất lượng theo NFE/FLOPs và thời gian. Dùng seed ghép cặp, khoảng tin cậy và kết quả theo từng nhóm; không chỉ báo trung bình gộp.

## 7. Xác minh và tiêu chí hoàn thành

1. **Dữ liệu và reward:** kiểm tra đủ các tệp prompt/ngữ pháp của ba nhóm, chống rò rỉ phân tử chuẩn hóa, chứng minh trường hợp được grammar nhận nhưng không thuộc nhóm theo SMARTS, kết quả grammar/RDKit/SMARTS trên các trường hợp dương và âm, token kết thúc và giới hạn độ dài.
2. **CARS:** trên một mô hình đồ chơi có thể liệt kê hết đầu ra, so khối xác suất trie, phân phối mẫu được chấp nhận và ước lượng $p_t\tau(Y_t)$ với đáp án chính xác. Kiểm tra trie không bao giờ loại một đầu ra có reward 1.
3. **Value trên MDLM:** tạo bộ trạng thái che test độc lập; đối chiếu CARS-value và learned verifier với ước lượng Monte Carlo có ngân sách lớn theo cùng $q_z$. Báo sai số, calibration và chi phí theo tỷ lệ vị trí bị che và theo nhóm. Kiểm tra ảnh hưởng của thứ tự mở token và sự khác biệt giữa $q_z$ với cách hoàn thiện gốc của MDLM.
4. **Tích hợp:** trên cùng trạng thái và seed, xác nhận MDM-VGB thường và phương pháp kết hợp chỉ khác nguồn value; reward terminal và bộ đếm chi phí không đổi. Kiểm tra kết quả luôn là SMILES có thể giải mã, kể cả khi lần chạy hết ngân sách.
5. **Thực nghiệm cuối:** tạo bảng/đường cong cho đủ ba phương pháp × ba nhóm × ba seed, kèm cấu hình, phiên bản dependency, checkpoint, dữ liệu và đầu ra thô để tái lập. Kết luận riêng về chất lượng phân tử, hiệu quả tính toán và độ chính xác của ước lượng value; nếu kiểm tra phân phối ở mục 4 thất bại, không tuyên bố đã ước lượng đúng $V^*$ lý tưởng.

## 8. Thứ tự thực hiện và điểm chặn

1. Khôi phục, chốt và kiểm định benchmark. **Chặn bước sau** nếu chưa có ví dụ/ngữ pháp/SMARTS đủ rõ để tính cùng reward cho cả ba phương pháp.
2. Chuẩn bị corpus, tokenizer và MDLM chung; xác nhận mô hình sinh được đầu ra đủ dài và không rò rỉ ví dụ benchmark.
3. Hoàn thành CARS trên chính sách MDLM và MDM-VGB với verifier học được; đo baseline riêng trước khi kết hợp.
4. Hoàn thành bộ ước lượng value dựa trên CARS; vượt qua kiểm tra mô hình đồ chơi và calibration trên trạng thái MDLM trước khi nối vào MDM-VGB.
5. Chạy ba phương pháp ở nhiều mức ngân sách, kiểm tra kết quả, xuất hiện vật và viết báo cáo.

Nguồn tham chiếu chính: `resource/2510.01902v2.pdf` (mục 4.2, Phụ lục H), `resource/2606.28301v1.pdf` (mục 3–5, Phụ lục F), `resource/CARS-code/` và `resource/MDM-VGB/`.
