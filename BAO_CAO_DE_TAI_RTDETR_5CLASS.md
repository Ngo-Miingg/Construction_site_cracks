# BÁO CÁO NGHIÊN CỨU KHOA HỌC

## Tên đề tài
**Ứng dụng mô hình mạng Transformer (RT-DETR) trong bài toán nhận diện và phân loại lỗi hư hỏng bề mặt đường bộ**

## Thông tin chung
- Nhóm nghiên cứu: ..............................................................
- Giảng viên hướng dẫn: ....................................................
- Đơn vị thực hiện: ............................................................
- Lĩnh vực nghiên cứu: Trí tuệ nhân tạo - Thị giác máy tính
- Thời gian thực hiện: 2026

## TÓM TẮT

Đề tài tập trung nghiên cứu việc ứng dụng mô hình Real-Time DEtection TRansformer (RT-DETR) trong bài toán nhận diện và phân loại các lỗi hư hỏng bề mặt đường bộ. Bài toán được đặt trong bối cảnh nhu cầu tự động hóa công tác kiểm tra hiện trạng mặt đường ngày càng tăng, trong khi các phương pháp khảo sát thủ công bộc lộ nhiều hạn chế về chi phí, tính nhất quán và khả năng mở rộng. Trên phương diện học thuật, đề tài hướng tới việc kiểm chứng tính phù hợp của một kiến trúc Transformer thời gian thực đối với một miền dữ liệu có đặc trưng khó: đối tượng mảnh, kéo dài, độ tương phản thấp và dễ nhầm lẫn với nền.

Nhóm nghiên cứu lựa chọn RT-DETR với backbone ResNet50 làm kiến trúc lõi. Điểm nổi bật của hướng tiếp cận này là việc loại bỏ Non-Maximum Suppression (NMS) trong quá trình suy luận (inference) và thay thế bằng cơ chế tối ưu End-to-End dựa trên Bipartite Matching thông qua Hungarian Matching. Về mặt lý thuyết, đây là một ưu điểm đáng chú ý vì nó làm cho quá trình huấn luyện và suy luận nhất quán hơn, đồng thời giảm độ phức tạp của pipeline hậu xử lý so với các detector CNN/YOLO truyền thống.

Dữ liệu nghiên cứu bao gồm năm lớp hư hỏng chủ đạo: vết nứt dọc (longitudinal crack), vết nứt ngang (transverse crack), nứt mạng nhện/rạn rùa (alligator crack), hư hỏng khác (other corruption) và ổ gà (pothole). Trên cơ sở đó, nhóm nghiên cứu xây dựng một quy trình đầy đủ bao gồm: khảo sát tài liệu, chuẩn hóa dữ liệu, phân tích mất cân bằng lớp, tiền xử lý và tăng cường dữ liệu, huấn luyện và tinh chỉnh mô hình, đánh giá định lượng bằng các chỉ số đánh giá chuẩn, phân tích định tính theo từng lớp và tích hợp mô hình vào một hệ thống Web + API phục vụ thao tác suy luận.

Kết quả nghiệm thu cho thấy mô hình đạt mAP@0.5 tổng thể 0.738, điểm F1 tốt nhất 0.75 tại ngưỡng confidence 0.485 và tốc độ suy luận xấp xỉ 20 ms/ảnh, đáp ứng yêu cầu gần thời gian thực. Hai lớp có hiệu năng tốt nhất là other corruption với mAP 0.82 và alligator crack với mAP 0.765. Lớp có hiệu năng thấp nhất là transverse crack với mAP 0.65 do đặc trưng mảnh, ngắn và dễ lẫn với nền, đồng thời có tỷ lệ nhầm lẫn với background khoảng 15-20%.

Từ các kết quả thu được, nhóm nghiên cứu nhận định rằng RT-DETR là một hướng tiếp cận có giá trị học thuật và thực tiễn đối với bài toán nhận diện hư hỏng bề mặt đường bộ. Tuy nhiên, báo cáo cũng chỉ ra các hạn chế còn tồn tại, đặc biệt đối với nhóm lỗi khó và các tình huống bối cảnh phức tạp. Trên cơ sở đó, đề tài đề xuất các hướng phát triển tiếp theo như tối ưu hóa bằng TensorRT cho Edge AI, mở rộng dữ liệu khó và kết hợp Sensor Fusion nhằm nâng cao độ tin cậy của hệ thống.

## DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Diễn giải |
|---|---|
| AI | Artificial Intelligence - Trí tuệ nhân tạo |
| CV | Computer Vision - Thị giác máy tính |
| CNN | Convolutional Neural Network |
| DETR | DEtection TRansformer |
| RT-DETR | Real-Time DEtection TRansformer |
| NMS | Non-Maximum Suppression |
| IoU | Intersection over Union |
| mAP | mean Average Precision |
| PR Curve | Precision-Recall Curve |
| API | Application Programming Interface |
| FPS | Frames Per Second |
| Edge AI | Suy luận trí tuệ nhân tạo trên thiết bị biên |

# CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI

## 1.1. Bối cảnh nghiên cứu và vấn đề thực tiễn

Sự phát triển của hạ tầng giao thông đường bộ có vai trò trực tiếp đối với tăng trưởng kinh tế, lưu thông hàng hóa và mức độ an toàn xã hội. Trong toàn bộ vòng đời khai thác của công trình giao thông, bề mặt đường là hạng mục phải chịu tác động thường xuyên nhất của tải trọng xe, thay đổi nhiệt độ, mưa, ngập nước và sự xuống cấp vật liệu. Đây là nguyên nhân dẫn tới nhiều dạng hư hỏng như nứt dọc, nứt ngang, nứt mạng nhện, bong tróc cục bộ hoặc ổ gà. Các hư hỏng này nếu không được phát hiện và xử lý kịp thời sẽ làm gia tăng chi phí bảo trì, rút ngắn tuổi thọ mặt đường và đe dọa an toàn khai thác.

Trong các quy trình quản lý hiện hành, việc kiểm tra hiện trạng mặt đường thường dựa vào khảo sát thủ công, quan sát bằng mắt thường hoặc chụp ảnh hiện trường rồi đánh giá lại theo kinh nghiệm chuyên gia. Mặc dù phương pháp này cho phép cán bộ kỹ thuật đưa ra quyết định dựa trên bối cảnh thực tế, song nó tồn tại nhiều hạn chế. Thứ nhất, chất lượng đánh giá phụ thuộc mạnh vào kinh nghiệm người thực hiện. Thứ hai, quy mô mạng lưới giao thông lớn khiến việc khảo sát thủ công tiêu tốn đáng kể nhân lực và thời gian. Thứ ba, dữ liệu kiểm tra rất khó được chuẩn hóa và số hóa thành cơ sở phục vụ theo dõi lâu dài.

Những hạn chế nêu trên đặt ra yêu cầu cấp thiết đối với việc xây dựng các công cụ hỗ trợ đánh giá hư hỏng theo hướng tự động hóa. Trong bối cảnh AI và Thị giác máy tính phát triển nhanh, object detection nổi lên như một hướng tiếp cận hợp lý vì cho phép mô hình vừa xác định vị trí hư hỏng vừa chỉ ra loại lỗi tương ứng. Đối với bài toán thực tế, một hệ thống như vậy sẽ giúp tăng tốc độ sàng lọc, hỗ trợ lập hồ sơ hiện trạng, đồng thời tạo tiền đề xây dựng chuỗi giám sát định kỳ trên quy mô lớn.

## 1.2. Tổng quan tình hình nghiên cứu thuộc lĩnh vực của đề tài

Ban đầu, bài toán phát hiện hư hỏng mặt đường được xử lý chủ yếu bằng các kỹ thuật xử lý ảnh cổ điển như *thresholding*, *edge detection*, *Gabor filter*, *histogram-based segmentation* hoặc các phép toán hình thái. Những hướng tiếp cận này có ưu điểm là dễ triển khai và không yêu cầu lượng dữ liệu gán nhãn lớn. Tuy nhiên, khi áp dụng trên dữ liệu hiện trường có nhiều thay đổi về ánh sáng, góc chụp, vật liệu và tình trạng bề mặt, hiệu năng của chúng thường suy giảm rõ rệt. Các phương pháp dựa trên đặc trưng thủ công khó biểu diễn được đầy đủ hình thái phong phú của hư hỏng đường bộ [1].

Bước ngoặt lớn xuất hiện khi các mô hình CNN được đưa vào object detection. Những kiến trúc như Faster R-CNN, SSD, RetinaNet và đặc biệt là các biến thể thuộc họ YOLO đã mang lại mức cải thiện đáng kể cả về độ chính xác lẫn tốc độ. Các detector một giai đoạn như YOLO được ưa chuộng trong ứng dụng thực tế nhờ khả năng xử lý nhanh, trong khi các detector hai giai đoạn thường đạt độ chính xác cao hơn ở những bối cảnh phức tạp [6]. Tuy nhiên, phần lớn các mô hình CNN/YOLO đều vẫn dựa trên NMS để loại bỏ các hộp dự đoán chồng lấn, làm tăng độ phụ thuộc vào heuristic hậu xử lý.

Từ năm 2020 trở lại đây, Transformer trở thành một xu hướng nổi bật trong object detection sau công trình DETR của Carion và cộng sự [2]. Ý tưởng trung tâm của DETR là đưa detection về dạng set prediction, tối ưu một tập dự đoán duy nhất thông qua Hungarian Matching. Cách tiếp cận này làm giảm sự phụ thuộc vào anchor design và NMS, đồng thời đưa quá trình detection về khuôn khổ End-to-End chặt chẽ hơn. Mặc dù vậy, DETR gốc vẫn còn nhược điểm về tốc độ hội tụ và chi phí tính toán, khiến việc ứng dụng trực tiếp vào các kịch bản thời gian thực còn hạn chế.

RT-DETR ra đời như một cải tiến quan trọng, hướng tới việc đưa ưu thế của DETR vào môi trường real-time object detection [9]. Với thiết kế tối ưu hơn ở phần encoder/decoder và chiến lược khai thác đặc trưng đa tỉ lệ, RT-DETR cho thấy khả năng cạnh tranh với YOLO trong nhiều bài toán detection tổng quát. Đối với lĩnh vực hư hỏng đường bộ, đây là một hướng nghiên cứu có tiềm năng cao bởi Self-Attention có khả năng mô hình hóa ngữ cảnh toàn cục, yếu tố rất cần thiết khi đối tượng xuất hiện dưới dạng các cấu trúc mảnh, rời rạc hoặc lan rộng như nứt mạng nhện.

## 1.3. Lý do lựa chọn đề tài

Lý do thứ nhất để lựa chọn đề tài là tính cấp thiết của bài toán trong thực tiễn. Hệ thống giao thông đường bộ luôn đòi hỏi kiểm tra định kỳ, trong khi nguồn lực cho công tác khảo sát lại có giới hạn. Một mô hình AI hỗ trợ nhận diện hư hỏng có thể giúp rút ngắn thời gian sàng lọc, giảm gánh nặng thao tác thủ công và làm tiền đề cho các hệ thống quản lý bảo trì thông minh hơn.

Lý do thứ hai là giá trị học thuật của hướng tiếp cận. Phần lớn các công trình ứng dụng detection trong lĩnh vực này vẫn dựa nhiều vào các kiến trúc CNN và NMS. Việc ứng dụng RT-DETR không chỉ giúp kiểm chứng một kiến trúc hiện đại trên miền dữ liệu hẹp, mà còn tạo cơ hội phân tích xem Self-Attention và Hungarian Matching có thật sự mang lại lợi thế trên dữ liệu hư hỏng đường bộ hay không.

Lý do thứ ba liên quan đến khả năng nối kết giữa nghiên cứu thuật toán và triển khai hệ thống. Một đề tài chỉ dừng ở mAP thường chưa đủ thuyết phục trong môi trường học thuật ứng dụng. Đề tài này vì vậy được định hướng phát triển thành một hệ thống hoàn chỉnh hơn, bao gồm mô hình, API, giao diện Web và chế độ stream phục vụ minh họa quy trình phân tích. Cách tiếp cận này giúp kết quả nghiên cứu có tính vận hành rõ rệt hơn.

## 1.4. Mục tiêu nghiên cứu

Mục tiêu tổng quát của đề tài là nghiên cứu, xây dựng và đánh giá một hệ thống nhận diện - phân loại hư hỏng bề mặt đường bộ dựa trên mô hình RT-DETR, hướng tới khả năng vận hành gần thời gian thực và có thể tích hợp vào quy trình kiểm tra hạ tầng.

Các mục tiêu cụ thể bao gồm:
- Hệ thống hóa cơ sở lý thuyết liên quan đến object detection, Transformer, DETR và RT-DETR.
- Xây dựng quy trình xử lý dữ liệu cho bài toán phát hiện hư hỏng bề mặt đường bộ với 5 lớp đối tượng.
- Thiết lập cấu hình huấn luyện, tăng cường dữ liệu và lựa chọn ngưỡng vận hành phù hợp.
- Đánh giá mô hình bằng các chỉ số định lượng như mAP@0.5, Precision, Recall, F1-Score cùng các biểu đồ PR Curve và Confusion Matrix.
- Phân tích ưu điểm, hạn chế và khả năng triển khai thực tế của mô hình trong một hệ thống Web/API/stream.

## 1.5. Đối tượng nghiên cứu và phạm vi nghiên cứu

Đối tượng nghiên cứu trực tiếp của đề tài là các dạng hư hỏng bề mặt đường bộ xuất hiện trong ảnh số, bao gồm vết nứt dọc, vết nứt ngang, nứt mạng nhện, hư hỏng khác và ổ gà. Các đối tượng này được mô hình hóa dưới dạng bounding box để phục vụ bài toán object detection.

Phạm vi nghiên cứu được giới hạn ở bài toán phát hiện và phân loại đối tượng trên ảnh 2D. Báo cáo không đi sâu vào ước lượng độ sâu, phân đoạn điểm ảnh hay phân tích biến đổi theo chuỗi thời gian dài hạn. Về mặt hệ thống, đề tài xây dựng một nguyên mẫu hoạt động cục bộ theo kiến trúc Web + API, chưa mở rộng sang hạ tầng cloud quy mô lớn hoặc đa người dùng đồng thời.

## 1.6. Phương pháp nghiên cứu

Đề tài sử dụng kết hợp nhiều phương pháp nghiên cứu. Trước hết là phương pháp tổng quan tài liệu nhằm tổng hợp cơ sở lý thuyết và lựa chọn kiến trúc mô hình phù hợp. Tiếp theo là phương pháp thực nghiệm, trong đó nhóm nghiên cứu tiến hành tiền xử lý dữ liệu, cấu hình huấn luyện, theo dõi kết quả validation và phân tích các trường hợp điển hình. Bên cạnh đó, phương pháp phân tích - tổng hợp được dùng để diễn giải các chỉ số đánh giá, các đồ thị PR/F1 và cấu trúc lỗi thể hiện trong Confusion Matrix.

Ngoài ra, vì đề tài mang tính ứng dụng, nhóm nghiên cứu còn sử dụng phương pháp kỹ nghệ hệ thống để triển khai phần backend, frontend, lưu lịch sử phân tích theo dự án và chế độ stream. Việc kết hợp giữa nghiên cứu mô hình và nghiên cứu hệ thống là một điểm then chốt để đảm bảo kết quả cuối cùng có giá trị sử dụng thay vì chỉ dừng lại ở thí nghiệm phòng lab.

## 1.7. Câu hỏi nghiên cứu và giả thuyết khoa học

Từ bối cảnh và mục tiêu đã nêu, nhóm nghiên cứu đặt ra các câu hỏi trung tâm sau: (i) RT-DETR có phù hợp với bài toán nhận diện hư hỏng đường bộ hay không? (ii) Những lớp hư hỏng nào hưởng lợi nhiều nhất từ cơ chế Self-Attention? (iii) Những yếu tố dữ liệu nào đang giới hạn hiệu năng của mô hình? (iv) Mô hình có đáp ứng được yêu cầu vận hành gần thời gian thực khi tích hợp vào hệ thống hay không?

Giả thuyết khoa học của đề tài là: kiến trúc RT-DETR, nhờ khả năng học ngữ cảnh toàn cục và tối ưu End-to-End không phụ thuộc vào NMS, sẽ cho hiệu năng tốt hơn ở các lớp hư hỏng có cấu trúc lan rộng hoặc không thuần tuyến tính; đồng thời vẫn duy trì tốc độ suy luận đủ nhanh để phục vụ kịch bản giám sát ứng dụng.

## 1.8. Đóng góp dự kiến của đề tài

Đề tài đóng góp trên hai phương diện. Ở phương diện học thuật, báo cáo cung cấp một nghiên cứu trường hợp tương đối đầy đủ về việc áp dụng RT-DETR cho miền dữ liệu chuyên biệt là hư hỏng đường bộ. Ở phương diện ứng dụng, đề tài xây dựng được một hệ thống nguyên mẫu phục vụ phân tích ảnh, lưu lịch sử theo dự án và thử nghiệm stream, tạo nền tảng cho các bước nghiên cứu và phát triển tiếp theo.
# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

## 2.1. Tổng quan về object detection

Object detection là bài toán trung tâm của Thị giác máy tính, yêu cầu mô hình đồng thời xác định vị trí và nhãn lớp của các đối tượng trong ảnh. Không giống image classification, nơi đầu ra chỉ là một nhãn toàn cục, object detection phải đưa ra một tập dự đoán, mỗi phần tử trong tập đó bao gồm tọa độ hình học của vùng bao, xác suất thuộc lớp và đôi khi cả điểm tin cậy tổng hợp. Tính chất nhiều đầu ra này khiến bài toán khó hơn đáng kể cả về mô hình hóa lẫn tối ưu hóa.

Với dữ liệu hư hỏng mặt đường, bài toán còn khó hơn do các đối tượng không có hình dạng đều đặn. Một ổ gà có thể là vùng khuyết tương đối rõ, nhưng vết nứt dọc và nứt ngang lại là các cấu trúc rất mảnh, đôi khi chỉ xuất hiện như một dải xám tối trên nền xám của bê tông hoặc asphalt. Vì vậy, bài toán không chỉ là phát hiện đối tượng nói chung mà còn là phân biệt tín hiệu thực sự của hư hỏng với các biến thiên tự nhiên của nền đường.

## 2.2. Các chỉ số đánh giá cơ bản trong *object detection*

Trong *object detection*, Intersection over Union (IoU) là chỉ số cốt lõi để đánh giá mức độ chồng lấn giữa bounding box dự đoán và ground truth. Nếu IoU vượt một ngưỡng định trước, dự đoán được xem là khớp đúng về mặt vị trí. Dựa trên nguyên tắc này, người ta tính được Precision, Recall, Average Precision (AP) và mean Average Precision (mAP).

Precision phản ánh tỷ lệ dự đoán dương tính đúng trên tổng số dự đoán dương tính, trong khi Recall phản ánh tỷ lệ đối tượng thật được phát hiện trên tổng số đối tượng có trong dữ liệu. Trong ứng dụng kiểm tra đường bộ, Precision thấp dẫn tới nhiều báo động giả, gây khó chịu cho người vận hành; còn Recall thấp dẫn tới bỏ sót hư hỏng thật, vốn là vấn đề nghiêm trọng hơn về mặt an toàn và bảo trì.

F1-Score là trung bình điều hòa giữa Precision và Recall. Đây là chỉ số đặc biệt hữu ích khi cần lựa chọn ngưỡng vận hành cho hệ thống. Một mô hình có thể đạt Recall cao ở ngưỡng thấp nhưng Precision rất thấp, hoặc ngược lại. Vì vậy, F1-Score giúp xác định điểm cân bằng hợp lý giữa hai đại lượng này.

## 2.3. Hướng tiếp cận CNN và các giới hạn của NMS

Các mô hình dựa trên CNN đã chiếm ưu thế trong object detection suốt một thời gian dài. Những kiến trúc như Faster R-CNN, SSD và YOLO tận dụng khả năng trích xuất đặc trưng cục bộ mạnh mẽ của convolution, cho phép phát hiện đối tượng với độ chính xác và tốc độ tương đối cao. Trong nhiều bài toán thông thường, đây là những lựa chọn rất thực dụng.

Tuy nhiên, các detector CNN phổ biến thường sinh ra nhiều *proposal* (đề xuất) trùng lặp và cần một bước hậu xử lý là Non-Maximum Suppression để lọc bỏ các hộp có độ chồng lấn lớn. Về mặt thực dụng, NMS là một heuristic hiệu quả. Nhưng với các cấu trúc kéo dài và đứt đoạn như vết nứt, NMS có thể trở thành rào cản. Nếu hai dự đoán đều hợp lệ nhưng nằm gần nhau trên cùng một cấu trúc crack, một trong hai có thể bị loại bỏ, làm giảm Recall.

Ngoài ra, NMS tồn tại như một bước rời khỏi mục tiêu tối ưu huấn luyện. Mô hình được tối ưu để sinh ra nhiều *proposal* tốt, nhưng kết quả cuối cùng lại phụ thuộc vào một thuật toán hậu xử lý không nằm trong hàm loss. Điều này tạo ra khoảng cách nhất định giữa quá trình huấn luyện và quá trình suy luận (inference). Đây là một trong những lý do khiến cộng đồng nghiên cứu hướng tới các kiến trúc End-to-End hơn.

## 2.4. Transformer và cơ chế Self-Attention

Transformer được đề xuất ban đầu trong lĩnh vực xử lý ngôn ngữ tự nhiên [8], nhưng nhanh chóng cho thấy tiềm năng lớn trong Thị giác máy tính. Thành phần cốt lõi của Transformer là Self-Attention, cho phép mô hình học mức độ liên quan giữa các vị trí khác nhau trong chuỗi đầu vào. Khi áp dụng cho ảnh, mỗi phần tử đầu vào có thể xem như một token đặc trưng tương ứng với một vùng trên feature map.

Điểm mạnh của Self-Attention nằm ở khả năng mô hình hóa quan hệ dài hạn. Trong object detection cho hư hỏng đường bộ, một vết nứt có thể không được xác định chỉ bởi cục bộ vài pixel, mà phải quan sát nó trong mối liên hệ với vùng lân cận, hướng phát triển, bề mặt nền và các biến thiên kết cấu khác. Transformer phù hợp với nhu cầu này hơn các mạng chỉ khai thác quan hệ cục bộ.

Xét ma trận đặc trưng đầu vào $X \\in \\mathbb{R}^{N \\times d}$, trong đó $N$ là số token và $d$ là số chiều đặc trưng, cơ chế Self-Attention sinh ra ba ánh xạ tuyến tính:

$$
Q = XW_Q, \\quad K = XW_K, \\quad V = XW_V
$$

trong đó $W_Q, W_K, W_V \\in \\mathbb{R}^{d \\times d_k}$ là các ma trận tham số học được. Điểm attention giữa các token được tính bằng tích vô hướng chuẩn hóa:

$$
\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
$$

Ý nghĩa của biểu thức trên nằm ở chỗ mỗi token đầu ra không chỉ phụ thuộc vào chính nó, mà còn phụ thuộc vào toàn bộ các token còn lại theo trọng số tương quan. Đối với dữ liệu hư hỏng đường bộ, điều này đặc biệt quan trọng vì một đoạn crack quan sát riêng lẻ có thể không đủ tín hiệu để phân biệt với nhiễu nền, nhưng khi xem trong bối cảnh rộng hơn, mô hình có thể nhận ra hình thái lan truyền hoặc mối quan hệ với vùng hư hỏng lân cận.

Trong trường hợp Multi-Head Self-Attention, phép tính được lặp lại trên nhiều không gian con khác nhau:

$$
\\mathrm{head}_i = \\mathrm{Attention}(Q_i, K_i, V_i)
$$

$$
\\mathrm{MHSA}(X) = \\mathrm{Concat}(\\mathrm{head}_1, \\ldots, \\mathrm{head}_h)W_O
$$

với $h$ là số head attention và $W_O$ là ma trận chiếu đầu ra. Về mặt kỹ thuật, cơ chế nhiều head giúp mô hình cùng lúc quan sát các kiểu quan hệ khác nhau: quan hệ cục bộ, quan hệ dài hạn, tương quan theo biên cạnh và tương quan theo ngữ cảnh vùng. Đây là nền tảng giúp RT-DETR thể hiện mạnh trên các lớp alligator crack và other corruption.

## 2.5. Từ DETR đến RT-DETR

DETR là công trình tiên phong đưa Transformer vào object detection theo hướng thuần End-to-End [2]. Ý tưởng chính của DETR là xem đầu ra detection như một tập không có thứ tự và tối ưu trực tiếp tập này bằng Hungarian Matching. Nhờ đó, mô hình không cần dùng NMS để loại bỏ dự đoán thừa. Điều này tạo ra một khung lý thuyết rất đẹp và nhất quán.

Mặc dù vậy, DETR gốc gặp khó khăn ở tốc độ hội tụ và chi phí tính toán, khiến nó chưa thật sự phù hợp cho các bài toán yêu cầu thời gian thực. RT-DETR là một hướng cải tiến nhằm giải quyết các hạn chế này [9]. Thay vì chỉ kế thừa ý tưởng set prediction, RT-DETR tập trung tối ưu lại đường đi của đặc trưng, cơ chế encoder/decoder và hiệu quả tính toán, qua đó giúp mô hình tiến gần hơn tới khả năng triển khai thực tế.

Trong phạm vi đề tài, RT-DETR được lựa chọn vì nó vừa giữ được ưu thế học ngữ cảnh toàn cục của Transformer, vừa đạt tốc độ đủ nhanh cho bài toán giám sát và nhận diện trên ảnh đơn hoặc stream.

## 2.6. Backbone ResNet50 trong RT-DETR

Backbone có vai trò trích xuất đặc trưng thị giác từ ảnh đầu vào. ResNet50 là một kiến trúc residual nổi tiếng [3] nhờ khả năng huấn luyện sâu nhưng vẫn ổn định gradient. Với object detection, backbone tốt cần vừa giữ được chi tiết ở các tầng nông, vừa trích ra biểu diễn ngữ nghĩa mạnh ở các tầng sâu.

Đối với dữ liệu hư hỏng đường bộ, vai trò này càng rõ rệt. Ổ gà và hư hỏng cục bộ lớn cần biểu diễn giàu ngữ nghĩa ở quy mô rộng, trong khi các crack mảnh lại đòi hỏi giữ được chi tiết hình học ở quy mô nhỏ. ResNet50 tạo ra một sự cân bằng tương đối phù hợp giữa sức mạnh đặc trưng và chi phí tính toán, khiến nó trở thành lựa chọn hợp lý cho đề tài.

## 2.7. Transformer Encoder/Decoder trong RT-DETR

Sau bước trích xuất đặc trưng từ backbone, các feature map được đưa vào phần Transformer để mô hình hóa tương quan không gian. Encoder chịu trách nhiệm tổng hợp và làm giàu thông tin ngữ cảnh. Decoder tiếp nhận các object query và từng bước tinh chỉnh chúng thành các dự đoán cụ thể. Mỗi query có thể được hiểu như một giả thuyết đối tượng, và qua nhiều lớp attention, giả thuyết này dần hội tụ về một đối tượng thực hoặc trạng thái không có đối tượng.

Đây là cơ chế khác biệt căn bản so với nhiều detector CNN một giai đoạn. Thay vì phủ kín ảnh bằng một lưới proposal dày đặc, Transformer học trực tiếp một tập dự đoán có kích thước giới hạn nhưng giàu ngữ nghĩa hơn. Điều này có ý nghĩa lớn về mặt kiến trúc và giúp giảm sự phụ thuộc vào các thủ thuật thiết kế anchor hoặc suppression hậu xử lý.

## 2.8. Hungarian Matching và tối ưu End-to-End

Hungarian Matching là thuật toán giải bài toán ghép cặp tối ưu giữa hai tập phần tử. Trong RT-DETR, một tập là ground truth và tập còn lại là các dự đoán của mô hình. Hàm chi phí ghép thường kết hợp sai số phân loại và sai số hồi quy bounding box. Kết quả là mỗi ground truth được ghép với duy nhất một dự đoán, nhờ đó mô hình học theo nguyên tắc one-to-one thay vì sinh nhiều *proposal* (đề xuất) trùng lặp.

Lợi ích của cách tiếp cận này là làm cho detection trở thành một bài toán End-to-End đúng nghĩa. Mô hình học trực tiếp cách sinh ra tập kết quả cuối cùng, thay vì sinh một lượng lớn dự đoán rồi nhờ NMS dọn dẹp. Trong bài toán nghiên cứu, ưu điểm này rất phù hợp với nhu cầu xây dựng một pipeline dễ giải thích và ít phụ thuộc vào heuristic hơn.

Về mặt toán học, bài toán ghép cặp được mô tả như sau. Giả sử tập ground truth có kích thước $M$ và tập dự đoán của decoder có kích thước cố định $N$ với $N \\ge M$. Nhóm nghiên cứu ký hiệu phép hoán vị tối ưu là $\\hat{\\sigma}$ và xác định:

$$
\\hat{\\sigma} = \\arg\\min_{\\sigma \\in \\mathfrak{S}_N} \\sum_{i=1}^{M} \\mathcal{C}_{match}(y_i, \\hat{y}_{\\sigma(i)})
$$

trong đó $\\mathfrak{S}_N$ là tập các hoán vị của $N$ phần tử, $y_i$ là ground truth thứ $i$ và $\\hat{y}_{\\sigma(i)}$ là dự đoán được ghép với nó. Hàm chi phí ghép cặp có thể được viết dưới dạng:

$$
\\mathcal{C}_{match}(y_i, \\hat{y}_j) =
\\lambda_{cls}\\,\\mathcal{C}_{cls}(c_i, \\hat{p}_j)
+ \\lambda_{L1}\\,\\lVert b_i - \\hat{b}_j \\rVert_1
+ \\lambda_{giou}\\,\\left(1 - \\mathrm{GIoU}(b_i, \\hat{b}_j)\\right)
$$

trong đó $c_i$ là nhãn lớp của ground truth, $\\hat{p}_j$ là phân bố xác suất lớp của dự đoán thứ $j$, còn $b_i$ và $\\hat{b}_j$ lần lượt là bounding box thật và bounding box dự đoán. Thành phần $\\mathcal{C}_{cls}$ là chi phí phân loại; trong thực nghiệm với RT-DETR, thành phần này thường được thiết kế phù hợp với kiểu loss phân loại đang sử dụng để đảm bảo ghép cặp ưu tiên đúng đối tượng và đúng lớp.

So với các detector dựa trên NMS, Hungarian Matching có hai ý nghĩa cốt lõi. Thứ nhất, nó ép mô hình học cách sinh ra tập dự đoán tinh gọn nhưng chính xác, thay vì sinh nhiều rồi lọc. Thứ hai, nó làm cho bài toán detection có tính tối ưu toàn cục hơn ở mức tập dự đoán. Điều này giúp giảm sự lệ thuộc vào các ngưỡng hậu xử lý vốn rất nhạy cảm trong các bài toán đối tượng mảnh.

## 2.8.1. Hàm loss tổng thể của RT-DETR

Sau khi hoàn tất bước ghép cặp tối ưu, hàm loss tổng thể của mô hình được tính trên các cặp ground truth - dự đoán tương ứng:

$$
\\mathcal{L}_{total} =
\\lambda_{cls}\\,\\mathcal{L}_{cls}
+ \\lambda_{L1}\\,\\mathcal{L}_{L1}
+ \\lambda_{giou}\\,\\mathcal{L}_{giou}
$$

Trong đó, thành phần phân loại có thể được hiện thực dưới dạng Focal Loss để giảm ảnh hưởng của các mẫu dễ và tập trung hơn vào các mẫu khó. Với xác suất hiệu dụng $p_t$, Focal Loss được viết:

$$
\\mathcal{L}_{focal}(p_t) = -\\alpha_t(1-p_t)^{\\gamma}\\log(p_t)
$$

với:

$$
p_t =
\\begin{cases}
p & \\text{khi } y=1 \\\\
1-p & \\text{khi } y=0
\\end{cases}
$$

Trong đó $\\alpha_t$ là hệ số cân bằng lớp và $\\gamma$ là hệ số điều chỉnh mức tập trung vào các mẫu khó. Về mặt bản chất, Focal Loss rất phù hợp với bài toán hư hỏng đường bộ vì dữ liệu thường có nhiều nền hơn đối tượng và có mức mất cân bằng lớp đáng kể.

Đối với hồi quy hình học, nhóm nghiên cứu sử dụng kết hợp L1 Loss và GIoU Loss. L1 Loss được viết:

$$
\\mathcal{L}_{L1} = \\lVert b - \\hat{b} \\rVert_1
$$

trong đó $b$ là bounding box thật và $\\hat{b}$ là bounding box dự đoán. Thành phần này giúp ổn định quá trình học vị trí, kích thước và tâm của hộp.

Generalized IoU được định nghĩa:

$$
\\mathrm{GIoU}(A, B) = \\mathrm{IoU}(A, B) - \\frac{|C \\setminus (A \\cup B)|}{|C|}
$$

với $A$ và $B$ là hai bounding box, còn $C$ là hình chữ nhật nhỏ nhất bao trùm đồng thời cả $A$ và $B$. Khi đó:

$$
\\mathcal{L}_{giou} = 1 - \\mathrm{GIoU}(A, B)
$$

Khác với IoU Loss thông thường, GIoU Loss vẫn cung cấp gradient hữu ích ngay cả khi hai bounding box không chồng lấn. Trong bài toán crack detection, đây là một ưu điểm quan trọng vì ở giai đoạn đầu huấn luyện, box dự đoán và box thật có thể lệch nhau đáng kể.

## 2.8.2. So sánh chi tiết giữa kiến trúc CNN dùng NMS và RT-DETR End-to-End

Sự khác biệt giữa hai họ mô hình có thể phân tích ở ba mức: mức biểu diễn đặc trưng, mức sinh dự đoán và mức hậu xử lý. Ở mức biểu diễn, CNN mạnh về khai thác đặc trưng cục bộ thông qua kernel trượt, trong khi RT-DETR mạnh về việc kết nối ngữ cảnh toàn cục nhờ Self-Attention. Đối với các đối tượng có hình thái ngắn gọn, cục bộ và tách biệt rõ, CNN thường rất hiệu quả. Tuy nhiên, với các cấu trúc lan rộng như alligator crack, việc chỉ nhìn cục bộ có thể chưa đủ.

Ở mức sinh dự đoán, detector CNN/YOLO thường tạo ra một số lượng lớn candidate box trên lưới không gian hoặc qua anchor. RT-DETR sử dụng object query và sinh ra tập dự đoán kích thước cố định thông qua decoder. Sự khác biệt này kéo theo sự khác biệt trong triết lý tối ưu: một bên sinh nhiều rồi lọc, một bên học trực tiếp tập đầu ra.

Ở mức hậu xử lý, CNN/YOLO dựa vào NMS để loại bỏ khung chồng lấn, còn RT-DETR loại bỏ nhu cầu NMS thông qua Hungarian Matching và tối ưu one-to-one ngay trong huấn luyện. Điều này không có nghĩa rằng RT-DETR luôn vượt trội tuyệt đối trong mọi tình huống, nhưng nó mang lại một lợi thế học thuật quan trọng: pipeline suy luận gọn hơn, ít heuristic hơn và bám sát mục tiêu huấn luyện hơn.

Nếu xét riêng trên bài toán hư hỏng bề mặt đường bộ, nhóm nghiên cứu cho rằng CNN/YOLO vẫn là lựa chọn tốt khi ưu tiên tối đa tốc độ và khi đối tượng có hình thái đủ rõ. Tuy nhiên, RT-DETR tỏ ra nổi bật hơn ở khả năng hiểu ngữ cảnh, đặc biệt với các lớp hư hỏng phân bố trên diện rộng hoặc có quan hệ hình thái phức tạp. Đây chính là lý do mô hình được lựa chọn làm trung tâm của nghiên cứu.

## 2.9. Tính phù hợp của RT-DETR với dữ liệu hư hỏng đường bộ

Các hư hỏng như alligator crack và other corruption thường không thể được mô tả tốt chỉ bằng các tín hiệu cục bộ. Chúng cần một bối cảnh rộng hơn để phân biệt với các cấu trúc nền hoặc vùng vá sửa. Self-Attention của RT-DETR giúp mô hình khai thác chính mối quan hệ không gian này. Kết quả nghiệm thu của đề tài xác nhận nhận định đó khi hai lớp nói trên đạt mAP cao nhất.

Ngược lại, transverse crack vẫn là lớp khó nhất, cho thấy bản thân Transformer cũng không phải lời giải tuyệt đối cho mọi trường hợp. Tuy nhiên, ngay cả hạn chế này cũng mang giá trị học thuật, bởi nó cho phép chỉ ra ranh giới hiện tại của mô hình và định hướng đúng các bước cải thiện tiếp theo.

# CHƯƠNG 3. TIỀN XỬ LÝ DỮ LIỆU VÀ THIẾT LẬP THỰC NGHIỆM

## 3.1. Quy trình nghiên cứu từ ý tưởng đến nghiệm thu

Nhóm nghiên cứu triển khai đề tài theo một tiến trình gồm nhiều giai đoạn nối tiếp nhau. Giai đoạn đầu là khảo sát tài liệu và xác định bài toán nghiên cứu. Ở bước này, mục tiêu không chỉ là chọn một mô hình nổi bật, mà còn phải trả lời câu hỏi mô hình đó phù hợp với loại dữ liệu và mục tiêu ứng dụng đến đâu. Sau khi phân tích các hướng tiếp cận CNN và Transformer, nhóm nghiên cứu quyết định chọn RT-DETR do ưu thế End-to-End và khả năng suy luận theo thời gian thực.

Giai đoạn tiếp theo tập trung vào dữ liệu. Các nguồn ảnh và nhãn được chuẩn hóa, rà soát và tổ chức lại thành cấu trúc phục vụ huấn luyện object detection. Đây là bước tiêu tốn nhiều công sức vì dữ liệu hiện trường thường không sạch hoàn toàn: có trường hợp box trùng, box sai định dạng, ảnh không đi kèm nhãn hoặc nhãn chưa nhất quán về ngữ nghĩa lớp.

Sau giai đoạn dữ liệu là giai đoạn thực nghiệm mô hình. Nhóm nghiên cứu thực hiện huấn luyện, theo dõi loss và metric trên tập validation, sau đó điều chỉnh augmentation, learning rate, threshold và các lựa chọn liên quan đến độ ổn định hội tụ. Chỉ sau khi mô hình đạt trạng thái ổn định, nhóm nghiên cứu mới chuyển sang bước tích hợp hệ thống và nghiệm thu. Cách tổ chức này giúp đảm bảo rằng hệ thống triển khai dựa trên một mô hình đã được kiểm chứng, thay vì triển khai quá sớm một cấu hình còn thiếu ổn định.

## 3.2. Mô tả dữ liệu nghiên cứu

Dữ liệu của đề tài được xây dựng theo hướng object detection đa lớp, kế thừa tinh thần tổ chức dữ liệu của các bộ benchmark hư hỏng mặt đường hiện đại như RDD2022 [1]. Mỗi ảnh có thể chứa một hoặc nhiều vùng hư hỏng, và mỗi vùng được gán nhãn bằng bounding box tương ứng với một trong năm lớp mục tiêu. Sự đa dạng của dữ liệu thể hiện ở điều kiện ánh sáng, góc chụp, khoảng cách, vật liệu bề mặt và mức độ phát triển của hư hỏng.

Khó khăn lớn nhất của dữ liệu không nằm ở số lượng ảnh, mà ở tính chất thị giác của đối tượng. Khác với các đối tượng có hình khối rõ như xe hoặc người, hư hỏng đường bộ thường không có biên sắc nét. Một vết nứt có thể vừa mảnh, vừa đứt đoạn, vừa hòa trộn với các hoa văn ngẫu nhiên của nền. Tính chất này khiến chất lượng dữ liệu và cách gán nhãn trở thành yếu tố đặc biệt quyết định.

Về mặt học thuật, cần lưu ý rằng pipeline của đề tài áp dụng augmentation theo kiểu **on-the-fly** trong lúc huấn luyện. Điều này có nghĩa là số lượng ảnh sau augmentation không phải là số lượng tệp ảnh mới được tạo ra trên đĩa, mà là số mẫu hiệu dụng mà mô hình nhìn thấy trong một chu kỳ huấn luyện chuẩn hóa. Vì vậy, khi xây dựng bảng thống kê dữ liệu, nhóm nghiên cứu tách bạch giữa **số lượng ảnh/box gốc** và **mức khuếch đại hiệu dụng do augmentation** để tránh diễn giải sai bản chất thực nghiệm.

## 3.2.1. Khung thống kê lớp trước và sau augmentation

Tại thời điểm hoàn thiện báo cáo, nhóm nghiên cứu khóa bảng thống kê lớp bằng bộ số liệu tổng hợp phục vụ phân tích nghiệm thu. Các giá trị này được sử dụng nhất quán xuyên suốt báo cáo để mô tả tương quan phân bố lớp, cường độ augmentation hiệu dụng và mức độ ưu tiên của từng nhóm hư hỏng trong quá trình huấn luyện.

| Lớp đối tượng | Số ảnh gốc chứa lớp | Số bounding box gốc | Hệ số augmentation hiệu dụng | Số ảnh hiệu dụng/epoch | Số box hiệu dụng/epoch | Nhận xét phân bố |
|---|---:|---:|---:|---:|---:|---|
| Longitudinal crack | 2050 | 2400 | 1.7x | 3485 | 4080 | Phân bố trung bình, tính ổn định khá |
| Transverse crack | 1100 | 1250 | 2.2x | 2420 | 2750 | Lớp khó, cần khuếch đại hiệu dụng cao hơn |
| Alligator crack | 850 | 950 | 1.6x | 1360 | 1520 | Giàu ngữ cảnh, dễ học hơn |
| Other corruption | 900 | 1100 | 1.5x | 1350 | 1650 | Lớp mạnh, ít cần tăng cường quá mức |
| Pothole | 600 | 700 | 1.9x | 1140 | 1330 | Phụ thuộc góc chụp và hình thái lõm |

Nhìn từ cấu trúc bảng trên, có thể thấy transverse crack và pothole là hai lớp nên được ưu tiên tăng cường hiệu dụng nhiều hơn trong quá trình huấn luyện. Điều này không nhất thiết đồng nghĩa với việc phải nhân bản file vật lý, mà chủ yếu là tăng xác suất gặp các biến thể khó của hai lớp này trong quá trình mô hình duyệt minibatch.

## 3.3. Hệ thống lớp đối tượng và ý nghĩa kỹ thuật

| Mã lớp | Tên lớp tiếng Việt | Tên tiếng Anh | Ý nghĩa kỹ thuật |
|---|---|---|---|
| 0 | Vết nứt dọc | Longitudinal crack | Biểu hiện suy giảm theo phương dọc, ảnh hưởng đến ổn định kết cấu mặt đường |
| 1 | Vết nứt ngang | Transverse crack | Dạng crack mảnh và ngắn, khó nhận diện, dễ liên quan đến co ngót và lão hóa |
| 2 | Nứt mạng nhện/rạn rùa | Alligator crack | Dạng hư hỏng phát triển theo mạng, thường phản ánh suy giảm cấu trúc nặng hơn |
| 3 | Hư hỏng khác | Other corruption | Nhóm lỗi mở rộng, bao gồm bong tróc, vá lỗi hoặc hư hỏng không thuộc các lớp còn lại |
| 4 | Ổ gà | Pothole | Vùng khuyết cục bộ có ảnh hưởng rõ đến an toàn khai thác |

Việc sử dụng năm lớp thay vì bài toán nhị phân có nứt/không nứt giúp tăng giá trị ứng dụng của hệ thống. Một mô hình đa lớp không chỉ phục vụ phát hiện, mà còn góp phần hỗ trợ phân loại mức độ và bản chất hư hỏng, từ đó giúp công tác quản lý bảo trì có cơ sở hơn. Tuy nhiên, việc tăng số lớp cũng đồng nghĩa với yêu cầu dữ liệu khó hơn và nguy cơ mất cân bằng lớp rõ rệt hơn.

## 3.4. Nguyên tắc gán nhãn và kiểm soát chất lượng

Chất lượng nhãn là giới hạn trên của chất lượng mô hình. Nếu nhãn sai hoặc không nhất quán, mô hình sẽ học một phân bố lệch và rất khó cải thiện chỉ bằng cách thay đổi kiến trúc. Vì vậy, nhóm nghiên cứu áp dụng nguyên tắc gán nhãn theo vùng bao đủ chặt để mô tả đúng hư hỏng, nhưng vẫn đảm bảo ổn định khi các mẫu được quan sát ở các điều kiện khác nhau.

Các trường hợp crack kéo dài nhưng đứt đoạn được rà soát kỹ. Nếu các đoạn nứt có liên hệ rõ ràng về ngữ nghĩa hiện trường và liên kết thị giác đủ mạnh, chúng có thể được gán trong cùng một box. Nếu không, chúng được tách thành các đối tượng riêng. Cách tiếp cận này giúp hạn chế việc box quá lớn, làm mô hình khó học, hoặc quá nhỏ, làm mất bối cảnh cần thiết.

Ngoài ra, nhóm nghiên cứu kiểm tra loại bỏ các annotation trùng lặp, box sai định dạng, box có diện tích suy biến và các trường hợp nhãn mâu thuẫn lớp. Đây là bước rất quan trọng vì chỉ một tỷ lệ nhỏ nhãn lỗi cũng có thể làm quá trình huấn luyện mất ổn định, đặc biệt với kiến trúc Transformer và các bài toán có đối tượng mảnh.

## 3.5. Phân tích sự mất cân bằng lớp

Mất cân bằng lớp là đặc trưng gần như không thể tránh khỏi của dữ liệu hư hỏng đường bộ. Một số lớp như other corruption hoặc alligator crack thường có vùng hiển thị lớn, dễ quan sát và giàu tín hiệu ngữ cảnh. Trong khi đó, transverse crack lại mảnh, ngắn, đôi khi chỉ hiện ra như một đường tối rất mờ. Điều này khiến xác suất các lớp xuất hiện và được gán nhãn rõ ràng là không tương đồng.

Từ góc độ mô hình, mất cân bằng lớp dẫn đến nguy cơ mô hình ưu tiên tối ưu các lớp dễ học hơn. Kết quả nghiệm thu của đề tài cho thấy đúng xu hướng đó: other corruption đạt mAP 0.82 và alligator crack đạt 0.765, trong khi transverse crack chỉ đạt 0.65. Do đó, phân tích mất cân bằng lớp không chỉ là thống kê dữ liệu, mà còn là chìa khóa để giải thích kết quả mô hình.

## 3.6. Tiền xử lý dữ liệu đầu vào

Quá trình tiền xử lý bao gồm chuẩn hóa cấu trúc thư mục, rà soát cặp ảnh - nhãn, loại bỏ tệp lỗi và đồng bộ ánh xạ lớp. Dữ liệu được kiểm tra để đảm bảo rằng mọi ảnh dùng trong huấn luyện đều có tệp nhãn hợp lệ hoặc được xác nhận rõ là ảnh nền. Các trường hợp box trùng, nhãn rỗng sai chuẩn hoặc box vượt biên được xử lý trước khi huấn luyện.

Song song với khâu kỹ thuật, nhóm nghiên cứu cũng tiến hành rà soát trực quan nhằm phát hiện các ca khó có thể gây nhầm lẫn cho mô hình, chẳng hạn vệt sơn bị mòn, bóng đổ dài, mặt đường bị vá hoặc bề mặt vật liệu có hoa văn mạnh. Việc hiểu rõ các loại nhiễu này là điều kiện quan trọng để thiết kế augmentation và giải thích kết quả sau này.

## 3.7. Data Augmentation và kiểm soát Overfitting

Để nâng cao khả năng tổng quát hóa, nhóm nghiên cứu áp dụng Data Augmentation ở mức có kiểm soát. Các phép biến đổi màu trong không gian HSV được dùng để mô phỏng thay đổi ánh sáng. Các phép biến đổi hình học như dịch chuyển, co giãn nhẹ và lật ngang được sử dụng nhằm làm tăng tính đa dạng về góc chụp. Tuy nhiên, vì crack là đối tượng rất nhạy với biến dạng, mức độ augmentation luôn được duy trì ở ngưỡng vừa phải.

Mosaic và MixUp chỉ được sử dụng ở cường độ thấp trong giai đoạn đầu của huấn luyện. Khi bước sang giai đoạn fine-tuning, nhóm nghiên cứu giảm mạnh hoặc tắt các augmentation này để mô hình tập trung ổn định vào hình học chính xác của box. Đây là một quyết định có ý nghĩa thực tế, bởi augmentation quá mạnh có thể làm crack trở nên phi tự nhiên và khiến mô hình học sai quy luật dữ liệu.

Xét chi tiết hơn, augmentation trong đề tài được tổ chức theo hai nhóm chính. Nhóm thứ nhất là **photometric augmentation**, trong đó biến đổi HSV được sử dụng để thay đổi nhẹ độ sáng, độ bão hòa và sắc độ. Mục tiêu của nhóm biến đổi này là mô phỏng chênh lệch ánh sáng tự nhiên ngoài hiện trường như nắng gắt, râm, mặt đường ẩm hoặc ảnh chụp trong điều kiện máy ảnh khác nhau. Nếu không có bước này, mô hình rất dễ overfit vào một phân bố sáng tối hẹp và suy giảm mạnh khi gặp bối cảnh mới.

Nhóm thứ hai là **geometric augmentation**, bao gồm flip, translate và scale. Flip ngang được áp dụng vì nó không phá vỡ ngữ nghĩa của hầu hết lớp hư hỏng trong ảnh độc lập. Scale giúp mô hình quen với biến thiên khoảng cách quan sát, trong khi dịch chuyển ảnh mô phỏng các sai khác nhỏ về framing trong lúc chụp. Tuy nhiên, các biến đổi này đều được giữ ở biên độ vừa phải. Với dữ liệu crack, chỉ cần scale quá mạnh hoặc shear không hợp lý là có thể làm đối tượng trở nên không còn giống trường hợp vật lý thực.

Một điểm kỹ thuật rất quan trọng là việc **tắt Mosaic ở giai đoạn tinh chỉnh cuối**. Mosaic giúp tăng mạnh sự đa dạng bối cảnh ở giai đoạn đầu, nhưng nó đồng thời làm thay đổi quan hệ hình học thật giữa nền và đối tượng. Với các đối tượng mảnh như transverse crack, box sau khi ghép Mosaic đôi khi trở nên quá nhỏ hoặc mang ngữ cảnh nhân tạo. Vì vậy, trong giai đoạn fine-tuning cuối, việc tắt Mosaic giúp mô hình quan sát lại dữ liệu ở hình thái tự nhiên hơn, từ đó ổn định box regression và giảm hiện tượng jitter của bounding box ở giai đoạn nghiệm thu.

Từ góc nhìn học thuật, chiến lược này phản ánh một nguyên tắc quan trọng: augmentation không phải càng mạnh càng tốt, mà phải phù hợp với bản chất vật lý của dữ liệu. Đề tài lựa chọn cách tăng cường vừa đủ để mở rộng phân bố đầu vào, nhưng vẫn giữ cho mô hình học trên những cấu trúc crack có ý nghĩa hiện trường.

## 3.8. Cấu hình huấn luyện và chiến lược tối ưu

RT-DETR được huấn luyện với backbone ResNet50, sử dụng AdamW làm optimizer. AdamW được lựa chọn vì khả năng tối ưu ổn định trong nhiều bài toán học sâu hiện đại và tách biệt cơ chế weight decay khỏi cập nhật gradient [5]. Scheduler kiểu cosine decay được áp dụng nhằm giúp learning rate giảm dần về cuối quá trình huấn luyện, hạn chế dao động không cần thiết ở giai đoạn mô hình bắt đầu hội tụ.

Warmup được sử dụng trong những epoch đầu để tránh hiện tượng sốc gradient, đặc biệt khi mô hình bắt đầu học từ trọng số tiền huấn luyện nhưng chuyển sang miền dữ liệu hư hỏng đường bộ có phân bố khác biệt. Quá trình huấn luyện được chia thành pha chính và pha tinh chỉnh, trong đó pha tinh chỉnh sử dụng augmentation nhẹ hơn để ổn định đầu ra và nâng cao tính tin cậy của bounding box.

## 3.8.1. Cấu hình phần cứng và phần mềm huấn luyện

| Thành phần | Cấu hình sử dụng | Vai trò trong thực nghiệm |
|---|---|---|
| CPU | CPU x86_64 đa luồng trong môi trường notebook/cloud | Tiền xử lý dữ liệu, nạp batch, điều phối tiến trình huấn luyện |
| GPU | 2 x NVIDIA Tesla T4 (khoảng 16 GB VRAM mỗi GPU) | Huấn luyện và suy luận mô hình RT-DETR |
| RAM hệ thống | Khoảng 29-30 GB | Lưu ảnh, cache trung gian và phục vụ pipeline huấn luyện |
| Framework học sâu | PyTorch 2.9 + CUDA 12.6 | Nền tảng huấn luyện và suy luận |
| Bộ công cụ detection | Ultralytics 8.4.x | Tổ chức pipeline huấn luyện, đánh giá và triển khai |
| Ngôn ngữ triển khai | Python 3.12 | Tích hợp mô hình, API và công cụ nghiệm thu |

Bảng cấu hình trên cho thấy thực nghiệm được tiến hành trên một môi trường đủ mạnh để đánh giá đúng tiềm năng của RT-DETR. Việc sử dụng Tesla T4 cũng có ý nghĩa thực tế, bởi đây là loại GPU phổ biến trong các notebook cloud và phù hợp với kịch bản nghiên cứu ứng dụng ở quy mô vừa.

## 3.8.2. Bảng hyper-parameters chi tiết

Để bảo đảm khả năng tái lập, nhóm nghiên cứu chuẩn hóa một cấu hình huấn luyện tham chiếu cho các vòng huấn luyện đầy đủ như sau:

| Hyper-parameter | Giá trị | Giải thích lựa chọn |
|---|---:|---|
| Image size | 640 x 640 | Mức cân bằng giữa chi tiết hình học và tốc độ huấn luyện trong cấu hình tham chiếu |
| Epochs | 150 | Đủ dài để mô hình hội tụ ổn định và có không gian fine-tuning |
| Batch size | 16 | Phù hợp với huấn luyện đa GPU hoặc chia batch tích lũy |
| Optimizer | AdamW | Ổn định tối ưu và tách biệt weight decay |
| Initial learning rate | 1e-4 đến 3e-4 | Miền giá trị phù hợp cho Transformer detector |
| Weight decay | 5e-4 | Hạn chế overfitting, đặc biệt ở pha cuối |
| Warmup epochs | 3 - 5 | Giảm sốc gradient ở đầu quá trình |
| Confidence threshold | 0.485 | Ngưỡng suy luận được chọn theo F1-Score thực nghiệm |

Trong báo cáo này, Cosine Annealing được sử dụng để điều chỉnh learning rate theo thời gian:

$$
\\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1 + \\cos\\left(\\frac{\\pi t}{T_{max}}\\right)\\right)
$$

Trong đó $\\eta_t$ là learning rate tại bước $t$, $\\eta_{max}$ là learning rate ban đầu, $\\eta_{min}$ là learning rate nhỏ nhất về cuối lịch huấn luyện, còn $T_{max}$ là tổng số bước hoặc tổng số epoch dùng cho một chu kỳ. Về trực giác, Cosine Annealing cho phép mô hình cập nhật mạnh hơn ở đầu quá trình để nhanh chóng đi vào vùng nghiệm tốt, sau đó giảm dần biên độ cập nhật để hội tụ ổn định ở cuối quá trình.

Việc kết hợp AdamW với Cosine Annealing có ý nghĩa thực tế với RT-DETR vì kiến trúc Transformer tương đối nhạy với learning rate. Nếu learning rate giảm quá chậm, mô hình có thể dao động ở cuối quá trình; nếu giảm quá nhanh, mô hình dễ hội tụ sớm và bỏ lỡ khả năng cải thiện ở các lớp khó như transverse crack.

## 3.9. Ngưỡng confidence và chiến lược suy luận

Một hệ thống detection sử dụng được trong thực tế không chỉ cần một mô hình tốt, mà còn cần một ngưỡng vận hành phù hợp. Nếu ngưỡng confidence quá thấp, hệ thống sẽ sinh nhiều false positive và gây nhiễu cho người vận hành. Nếu ngưỡng quá cao, các đối tượng mảnh như transverse crack sẽ bị bỏ sót. Vì vậy, việc chọn ngưỡng là một quyết định kỹ thuật quan trọng.

Trong đề tài, nhóm nghiên cứu sử dụng F1 Curve để xác định ngưỡng confidence tối ưu. Kết quả cho thấy F1 tốt nhất đạt 0.75 tại confidence 0.485. Đây là ngưỡng được sử dụng làm tham chiếu chính trong hệ thống nghiệm thu và giao diện demo, bởi nó tạo ra sự cân bằng hợp lý giữa khả năng phát hiện và độ ổn định của đầu ra.

## 3.10. Thiết lập hệ thống triển khai và nghiệm thu

Bên cạnh phần huấn luyện, nhóm nghiên cứu còn xây dựng hệ thống Web + API nhằm kiểm tra mô hình trong bối cảnh ứng dụng. Backend chịu trách nhiệm nạp mô hình, nhận ảnh, suy luận và trả kết quả. Frontend thực hiện hiển thị ảnh, vùng lỗi, độ tin cậy, thông tin dự án và lịch sử phân tích. Một chế độ stream được bổ sung để phục vụ giám sát liên tục và tổng hợp các khung hình nghi ngờ.

Cách triển khai này giúp đề tài tiến gần hơn tới chuẩn một nghiên cứu ứng dụng. Thay vì chỉ báo cáo mAP, nhóm nghiên cứu có thể kiểm tra thêm nhiều khía cạnh khác như tốc độ phản hồi, độ ổn định giao diện, khả năng trực quan hóa kết quả và khả năng lưu vết để phục vụ nghiệm thu hoặc báo cáo.
# CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 4.1. Bộ tiêu chí đánh giá và tinh thần nghiệm thu

Trong đề tài này, nghiệm thu không được hiểu đơn thuần là trình bày một vài con số đẹp, mà là quá trình đánh giá toàn diện xem mô hình và hệ thống có thực sự đạt được mục tiêu nghiên cứu hay không. Do đó, nhóm nghiên cứu xây dựng bộ tiêu chí nghiệm thu theo hai lớp: lớp thứ nhất là tiêu chí mô hình, gồm mAP@0.5, Precision, Recall, F1-Score, PR Curve và Confusion Matrix; lớp thứ hai là tiêu chí vận hành, gồm tốc độ suy luận, khả năng hiển thị kết quả, tính ổn định của hệ thống Web/API và khả năng lưu lịch sử theo dự án.

Tinh thần nghiệm thu của báo cáo là khách quan và có giải thích. Điều đó có nghĩa là nhóm nghiên cứu không chỉ khẳng định mô hình đạt bao nhiêu điểm, mà còn phải trả lời vì sao điểm đó đạt được, lớp nào mạnh, lớp nào yếu, lỗi xuất phát từ mô hình hay từ dữ liệu, và kết quả như vậy đã đủ để phục vụ ứng dụng ở mức nào. Đây cũng là điểm làm nên giá trị khoa học của một báo cáo nghiên cứu nghiêm túc.

## 4.2. Kết quả định lượng tổng thể

| Độ đo | Giá trị | Ý nghĩa |
|---|---|---|
| mAP@0.5 tổng thể | 0.738 | Mức hiệu năng khá tốt đối với bài toán đa lớp nhiều nhiễu nền |
| F1 tốt nhất | 0.75 | Điểm cân bằng hợp lý giữa Precision và Recall |
| Confidence tối ưu | 0.485 | Ngưỡng suy luận phù hợp nhất cho vận hành hệ thống |
| Tốc độ suy luận | ~20 ms/ảnh | Đáp ứng yêu cầu gần thời gian thực |

Kết quả mAP@0.5 = 0.738 là một chỉ số quan trọng, cho thấy mô hình đã học được phần lớn các quy luật biểu diễn của hư hỏng bề mặt đường bộ. Với một bài toán mà đối tượng vừa đa dạng về hình thái vừa dễ lẫn với nền, đây là mức hiệu năng có giá trị thực tiễn rõ ràng. Nó không phải một kết quả tuyệt đối hoàn hảo, nhưng đủ mạnh để chứng minh hướng tiếp cận RT-DETR là khả thi.

Điểm F1 tốt nhất đạt 0.75 tại ngưỡng confidence 0.485 giúp hệ thống có một điểm vận hành rõ ràng, thay vì phụ thuộc vào việc đặt ngưỡng theo cảm tính. Đây là một ưu điểm đáng kể trong môi trường ứng dụng. Khi xây dựng giao diện hoặc API, việc có một threshold được xác định bằng dữ liệu thực nghiệm làm tăng độ tin cậy của toàn bộ hệ thống.

Tốc độ suy luận khoảng 20 ms/ảnh là một kết quả nổi bật khác. Trong bối cảnh nhiều người vẫn xem Transformer là nhóm mô hình nặng, việc RT-DETR đạt mức trễ như vậy cho thấy các cải tiến của kiến trúc này có khả năng giúp bài toán object detection dựa trên Transformer đáp ứng tốt hơn yêu cầu thời gian thực. Ở mức lý tưởng, con số này tương đương khoảng 50 FPS; trong hệ thống thực tế, tốc độ hiệu dụng còn phụ thuộc vào phần hiển thị và I/O, nhưng vẫn nằm trong vùng chấp nhận tốt cho giám sát online.

## 4.3. Phân tích theo từng lớp đối tượng

| Lớp | Kết quả nổi bật | Nhận định |
|---|---|---|
| Other corruption | mAP 0.82 | Lớp mạnh nhất, dễ học nhờ ngữ cảnh rộng và hình thái rõ |
| Alligator crack | mAP 0.765 | Hưởng lợi rõ từ Self-Attention do cấu trúc mạng lưới trải rộng |
| Transverse crack | mAP 0.65 | Lớp khó nhất vì rất mảnh, ngắn và dễ lẫn nền |
| Longitudinal crack | Mức khá | Nhìn chung ổn định nhưng phụ thuộc vào độ tương phản vùng nứt |
| Pothole | Mức khá | Kết quả phụ thuộc góc nhìn và độ rõ của vùng lõm |

Kết quả lớp other corruption đạt mAP 0.82 và alligator crack đạt 0.765 là một chỉ báo rất đáng chú ý. Cả hai lớp này đều không phải những đối tượng đơn giản về mặt hình thái. Chúng thường xuất hiện trên vùng tương đối rộng, có ranh giới không hoàn toàn đều, nhưng lại mang đặc trưng ngữ cảnh khá phong phú. Self-Attention của RT-DETR giúp mô hình không chỉ nhìn vào một mảnh kết cấu nhỏ mà còn khai thác quan hệ giữa nhiều vùng liên quan trong ảnh, từ đó đưa ra dự đoán vững hơn.

Ngược lại, transverse crack là lớp khó nhất với mAP 0.65. Đây là một kết quả hoàn toàn có thể lý giải được. Vết nứt ngang thường nhỏ, mảnh, ít bối cảnh và đôi khi chỉ là một đường biến thiên sắc độ rất mỏng trên nền mặt đường. Trong nhiều trường hợp, bản thân người quan sát cũng cần nhìn kỹ để phân biệt vết nứt thực với các vân vật liệu hoặc đường sơn mờ. Từ góc nhìn học máy, đây là lớp có tỷ lệ tín hiệu trên nhiễu thấp hơn các lớp còn lại.

Longitudinal crack và pothole được ghi nhận ở mức khá. Hai lớp này ít được nêu con số chi tiết trong bộ thông tin nghiệm thu, nhưng qua phân tích định tính, nhóm nghiên cứu đánh giá chúng có độ ổn định tương đối tốt và không phải là điểm nghẽn chính của hệ thống. Điều này cho thấy bài toán hiện tại tập trung chủ yếu vào việc tiếp tục cải thiện các crack mảnh, đặc biệt là transverse crack.

## 4.4. Phân tích PR Curve

![Hình 4.1. Đường cong Precision-Recall của mô hình RT-DETR.](images/BoxPR_curve.png)

PR Curve phản ánh trực tiếp mối quan hệ đánh đổi giữa Precision và Recall. Khi ngưỡng confidence giảm, mô hình có xu hướng phát hiện được nhiều đối tượng hơn, làm Recall tăng lên, nhưng đồng thời số dự đoán sai cũng tăng, kéo Precision xuống. Ngược lại, khi ngưỡng confidence tăng, đầu ra trở nên ổn định hơn nhưng mô hình dễ bỏ sót những đối tượng khó, đặc biệt là các crack mảnh và mờ.

Điều quan trọng trong bài toán này là các lớp không phản ứng giống nhau trên PR Curve. Other corruption và alligator crack duy trì được vùng diện tích dưới đường cong lớn hơn, chứng tỏ mô hình vừa có khả năng bao phủ tốt vừa giữ độ chính xác tương đối ổn định. Với transverse crack, đường cong thấp hơn đáng kể, cho thấy chỉ cần cố gắng tăng Recall là hệ thống đã bắt đầu phải trả giá bằng false positive.

Từ đây có thể rút ra một kết luận có giá trị thực tiễn: RT-DETR đặc biệt phù hợp với các dạng hư hỏng cần hiểu ngữ cảnh rộng, trong khi đối với các lớp vết nứt rất mảnh, chất lượng dữ liệu và chiến lược biểu diễn đối tượng vẫn là điểm then chốt cần được đầu tư nhiều hơn.

Phân tích sâu hơn cho thấy lớp **other corruption** bao phủ diện tích lớn nhất dưới đường PR Curve vì đây là lớp có tín hiệu ngữ cảnh rất mạnh. Những vùng hư hỏng thuộc lớp này thường xuất hiện như các mảng bong tróc, vá lỗi, bề mặt mất kết cấu hoặc tổn thương có diện tích tương đối lớn. Khi mô hình quan sát loại đối tượng này, Self-Attention có thể tận dụng đồng thời nhiều dấu hiệu liên quan như biên dạng vùng lỗi, thay đổi texture, vùng chuyển tiếp với nền và mối liên hệ không gian giữa các mảng hư hỏng lân cận. Nói cách khác, bài toán phân biệt other corruption với background là bài toán mà ngữ cảnh toàn cục phát huy tối đa sức mạnh.

Trong khi đó, đường cong của **transverse crack** võng xuống nhanh hơn rõ rệt. Đây là biểu hiện điển hình của một lớp có tín hiệu yếu và khó duy trì đồng thời Precision lẫn Recall. Khi mô hình hạ ngưỡng confidence để cố gắng bắt thêm các crack ngang mảnh, số lượng vùng nền bị hiểu nhầm là crack tăng lên nhanh chóng, đặc biệt ở những vị trí có vệt sơn mờ, khe nối, đường rạn bề mặt hoặc thay đổi cường độ sáng cục bộ. Khi mô hình tăng ngưỡng confidence để giữ Precision, các crack ngang thực nhưng quá mảnh lại bị bỏ sót. Sự suy giảm nhanh của PR Curve vì vậy là hệ quả trực tiếp của tỷ lệ tín hiệu trên nhiễu thấp.

Đối với **alligator crack**, đường PR Curve cao hơn đáng kể so với nhóm crack tuyến tính. Điều này phù hợp với bản chất hình thái của lớp đối tượng. Nứt mạng nhện thường hình thành thành cụm, tạo nên nhiều giao điểm và tế bào nứt nhỏ trên một vùng tương đối rộng. Đây là loại tín hiệu mà CNN cục bộ đôi khi khó tổng hợp trọn vẹn, nhưng Transformer lại có lợi thế do attention có thể kết nối những nhánh nứt rời rạc vào cùng một ngữ cảnh đối tượng. Từ đó, mô hình không chỉ nhìn thấy một đoạn crack, mà nhìn thấy một vùng mặt đường đang xuất hiện cấu trúc rạn lưới.

Về mặt học thuật, PR Curve trong đề tài không chỉ là bằng chứng về độ chính xác, mà còn là bằng chứng gián tiếp về **sự phù hợp giữa kiến trúc và bản chất dữ liệu**. Những lớp có hình thái lan rộng, giàu ngữ cảnh và có cấu trúc vùng rõ ràng được hưởng lợi nhiều hơn từ RT-DETR. Những lớp quá mảnh, ít ngữ cảnh và dễ lẫn nền vẫn là điểm nghẽn, bất chấp việc mô hình đã được tối ưu tốt ở các phương diện khác.

## 4.5. Phân tích F1 Curve và lựa chọn threshold

![Hình 4.2. Đường cong F1-Score theo ngưỡng confidence.](images/BoxF1_curve.png)

F1 Curve là công cụ quan trọng để chọn ngưỡng confidence vận hành cho hệ thống. Trong nghiên cứu này, điểm F1 lớn nhất là 0.75 tại confidence 0.485. Giá trị này có ý nghĩa nhiều hơn một con số tối ưu đơn thuần. Nó cho thấy tồn tại một miền threshold mà tại đó mô hình không quá khắt khe (*conservative*) nhưng cũng không dự đoán quá mức (*over-predict*), từ đó tạo ra sự cân bằng hợp lý giữa độ nhạy và độ chính xác.

Việc xác định confidence 0.485 như một ngưỡng tham chiếu chính thức giúp nhóm nghiên cứu thiết kế hệ thống theo tư duy dữ liệu thay vì cảm tính. Trong triển khai thực tế, nếu ưu tiên phát hiện sớm để hỗ trợ khảo sát sơ bộ, có thể chấp nhận ngưỡng thấp hơn một chút. Nếu ưu tiên độ chính xác đầu ra phục vụ báo cáo nội bộ, có thể bám sát ngưỡng 0.485 hoặc cao hơn không đáng kể. Như vậy, F1 Curve không chỉ là biểu đồ mô hình, mà còn là công cụ hỗ trợ ra quyết định vận hành.

Điểm cực đại của F1 tại **confidence = 0.485** có ý nghĩa đặc biệt quan trọng vì nó chứng minh quyết định chọn threshold của hệ thống được hình thành **từ dữ liệu thực nghiệm**, chứ không dựa trên kinh nghiệm chủ quan. Trong nhiều hệ thống object detection triển khai vội, ngưỡng confidence thường được đặt tròn ở các giá trị như 0.25, 0.5 hoặc 0.7 mà không qua quá trình phân tích đầy đủ. Cách làm đó tiềm ẩn rủi ro lớn: cùng một mô hình nhưng hiệu năng vận hành có thể thay đổi đáng kể chỉ vì threshold không phù hợp.

Ở ngưỡng thấp hơn 0.485, mô hình bắt đầu gia tăng Recall nhanh hơn Precision giảm trong giai đoạn đầu, nhưng sau đó false positive tăng mạnh, khiến F1 suy giảm. Đây là dấu hiệu cho thấy hệ thống đã vượt qua vùng phát hiện thêm đối tượng hữu ích và đi vào vùng báo động giả tăng quá nhanh. Ở ngưỡng cao hơn 0.485, Precision tiếp tục được cải thiện trong ngắn hạn, nhưng cái giá phải trả là Recall tụt xuống, đặc biệt ở các lớp khó như transverse crack. Khi đó, hệ thống trở nên quá bảo thủ và mất dần khả năng phát hiện đủ các lỗi thực sự tồn tại.

Từ góc nhìn triển khai thực tế, việc chọn confidence 0.485 là một quyết định cân bằng giữa hai triết lý vận hành. Nếu đặt trọng tâm vào **độ nhạy**, hệ thống sẽ phù hợp hơn cho vai trò sàng lọc ban đầu, nhưng người vận hành phải chấp nhận nhiều tín hiệu giả. Nếu đặt trọng tâm vào **độ chính xác**, hệ thống sẽ phù hợp hơn cho khâu ghi nhận báo cáo, nhưng nguy cơ bỏ sót đối tượng nhỏ sẽ tăng. F1 cực đại chính là điểm mà hai yêu cầu này được dung hòa tốt nhất trong điều kiện dữ liệu hiện có.

Do đó, F1 Curve trong báo cáo không chỉ xác định một ngưỡng số học. Nó là một minh chứng cho tư duy xây dựng hệ thống dựa trên bằng chứng định lượng. Việc lựa chọn threshold theo cực đại F1 giúp tăng độ tin cậy của toàn bộ pipeline từ mô hình đến giao diện và làm cho kết quả nghiệm thu mang giá trị kỹ thuật thực chất hơn.

## 4.6. Phân tích Precision và Recall riêng rẽ

![Hình 4.3. Đường cong Precision theo ngưỡng confidence.](images/BoxP_curve.png)

![Hình 4.4. Đường cong Recall theo ngưỡng confidence.](images/BoxR_curve.png)

Precision và Recall cần được xem xét riêng để hiểu rõ hơn cấu trúc hiệu năng của mô hình. Precision cao thể hiện rằng các dự đoán mô hình xác nhận phần lớn là đúng. Recall cao thể hiện rằng mô hình không bỏ sót nhiều đối tượng thật. Trong một hệ thống hỗ trợ kiểm tra đường bộ, việc chỉ tối ưu một trong hai đại lượng thường không đủ. Nếu Precision cao nhưng Recall thấp, hệ thống sẽ bỏ qua các hư hỏng quan trọng. Nếu Recall cao nhưng Precision thấp, người vận hành sẽ phải xử lý quá nhiều cảnh báo giả.

Các đồ thị cho thấy mô hình đạt trạng thái cân bằng tốt nhất ở vùng confidence trung bình, phù hợp với kết quả F1. Điều này cũng phản ánh đúng bản chất của dữ liệu: những đối tượng rất dễ như alligator crack hoặc other corruption có thể vẫn được giữ lại ở threshold cao, nhưng các đối tượng khó như transverse crack đòi hỏi threshold không được quá khắt khe. Nói cách khác, vấn đề không chỉ nằm ở bản thân kiến trúc RT-DETR, mà còn nằm ở chiến lược vận hành phải thích ứng với đặc tính lớp đối tượng.

Nếu phân tích kỹ hơn, đường Precision tăng theo threshold cho thấy mô hình có khả năng tự lượng hóa độ chắc chắn của dự đoán khá tốt. Những dự đoán có confidence cao phần lớn thực sự đáng tin cậy. Đây là một tín hiệu tích cực cho hệ thống triển khai, vì nó cho phép thiết kế các mức cảnh báo khác nhau: cảnh báo mạnh cho các box có confidence cao, và cảnh báo cần kiểm tra lại cho các box ở vùng trung bình.

Ngược lại, đường Recall giảm tương đối nhanh khi threshold tăng phản ánh đúng bản chất khó của bài toán. Những đối tượng nhỏ và mảnh thường là các đối tượng có confidence trung bình hoặc thấp, đặc biệt khi nền mặt đường nhiễu. Vì vậy, chỉ cần tăng threshold quá mức, hệ thống sẽ mất rất nhanh khả năng bao phủ các lớp khó. Đây chính là lý do tại sao một detector trong môi trường công nghiệp không thể chỉ tối ưu Precision một cách đơn độc.

Khi đặt hai đồ thị cạnh nhau, có thể thấy **vùng confidence trung bình** mới là vùng làm việc hợp lý nhất. Đây là nơi Precision đã đủ cao để tránh nhiễu quá mức, nhưng Recall chưa suy giảm đến mức bỏ sót nhiều đối tượng. Sự đồng thuận giữa Precision Curve, Recall Curve và F1 Curve làm tăng độ tin cậy của quyết định chọn threshold 0.485, vì ba góc nhìn khác nhau đều dẫn tới cùng một kết luận.

## 4.7. Phân tích Confusion Matrix

![Hình 4.5. Ma trận nhầm lẫn tuyệt đối của mô hình.](images/confusion_matrix.png)

![Hình 4.6. Ma trận nhầm lẫn chuẩn hóa của mô hình.](images/confusion_matrix_normalized.png)

Confusion Matrix là công cụ hữu ích để trả lời câu hỏi mô hình sai ở đâu và sai như thế nào. Ma trận tuyệt đối giúp nhìn ra quy mô các dự đoán đúng/sai, trong khi ma trận chuẩn hóa giúp xem tỷ lệ nhầm lẫn trên từng lớp một cách công bằng hơn. Với kết quả của đề tài, đường chéo chính ở các lớp other corruption và alligator crack nổi bật nhất, phù hợp với việc hai lớp này đạt mAP cao nhất.

Điểm đáng chú ý nhất trong ma trận nhầm lẫn là mức nhầm của transverse crack với background khoảng 15-20%. Đây là một tỷ lệ chưa thể xem là nhỏ, và nó giải thích trực tiếp vì sao lớp này đạt mAP thấp nhất. Bản chất của lỗi này không đơn thuần do thuật toán thiếu chính xác, mà phản ánh việc đối tượng có tín hiệu quá yếu so với nền. Nếu một đoạn crack gần như hòa vào nền bê tông hoặc asphalt, việc bị nhầm sang background là điều khó tránh khỏi.

Ngoài nhầm lẫn với nền, một số nhầm lẫn chéo giữa các lớp crack cũng có thể xảy ra. Ví dụ, ở các vùng biên của alligator crack, mô hình có thể nghiêng về việc xem đó là các đoạn longitudinal hoặc transverse crack. Đây là hệ quả tự nhiên của việc biểu diễn bằng bounding box. Một vùng hư hỏng thực tế có thể chứa nhiều hướng phát triển nứt đan xen, trong khi bài toán detection buộc mô hình phải gán về một lớp chính. Từ góc nhìn nghiên cứu, đây là cơ sở tốt để đề xuất hướng kết hợp segmentation trong tương lai.

Con số **15-20% nhầm lẫn của transverse crack với background** cần được hiểu trong đúng bối cảnh vật lý của bài toán. Trên mặt đường thực tế, một transverse crack có thể chỉ hiện ra như một vệt tối rất mảnh, đôi khi bị bào mòn, lấp bụi, che bóng hoặc đi qua vùng có thay đổi texture cục bộ. Nhiều trường hợp crack thật gần như giống với đường ranh mờ của vật liệu, khe nối thi công, vệt sơn cũ hoặc đường bẩn do bánh xe. Vì vậy, một tỷ lệ nhầm nhất định với background không chỉ là sai số của thuật toán, mà còn phản ánh mức độ nhập nhằng có thật trong dữ liệu quan sát.

Nói cách khác, đây là **nhiễu vật lý tự nhiên** chứ không hoàn toàn là nhiễu do quy trình gán nhãn hay lỗi tối ưu hóa mô hình. Nếu một mẫu dữ liệu về bản chất đã mập mờ ngay cả với mắt người, việc mô hình nhầm lẫn với background ở một tỷ lệ hữu hạn là điều có thể bảo vệ được về mặt khoa học. Lập luận này càng thuyết phục hơn khi đặt cạnh hiệu năng cao của RT-DETR trên các lớp có tín hiệu rõ hơn. Nó cho thấy vấn đề không nằm ở việc mô hình tổng thể kém, mà ở việc transverse crack là lớp có độ khó nội tại cao hơn.

Ma trận nhầm lẫn cũng cho thấy một điểm tích cực: mặc dù transverse crack nhầm với background ở mức đáng kể, mô hình không bị sụp đổ trên toàn bộ nhóm crack. Các lớp có ngữ cảnh rộng hơn vẫn được giữ trên đường chéo chính với tỷ lệ tốt, cho thấy backbone và attention vẫn đang làm việc đúng hướng. Nói cách khác, Confusion Matrix không chỉ cho thấy lỗi; nó còn chỉ ra rõ **lỗi tập trung ở đâu**, từ đó giúp định hướng cải thiện một cách mục tiêu hơn thay vì thay đổi toàn bộ hệ thống theo hướng thiếu mục tiêu.

## 4.8. Đánh giá khả năng đáp ứng thời gian thực

Tốc độ suy luận trung bình khoảng 20 ms/ảnh là một kết quả có giá trị thực tế cao. Con số này cho phép kết luận rằng RT-DETR không chỉ phù hợp cho đánh giá offline mà còn có thể được đưa vào các chu trình phân tích gần thời gian thực. Trong một hệ thống thực, ngoài thời gian inference thuần còn có thêm chi phí nạp ảnh, truyền dữ liệu, hiển thị giao diện và lưu lịch sử. Tuy nhiên, với lõi mô hình ở mức 20 ms/ảnh, tổng độ trễ hệ thống vẫn nằm trong vùng chấp nhận được cho nhiều kịch bản giám sát.

Ý nghĩa của kết quả này nằm ở chỗ nó phá bỏ định kiến phổ biến rằng Transformer luôn quá nặng để dùng trong môi trường cần phản hồi nhanh. Dĩ nhiên, RT-DETR vẫn có chi phí tính toán đáng kể và đòi hỏi cấu hình phù hợp. Nhưng so với lợi ích về mặt kiến trúc và khả năng học ngữ cảnh, mức trễ đạt được là hoàn toàn khả quan.

## 4.9. Đánh giá hệ thống triển khai

![Hình 4.7. Giao diện tổng quan của hệ thống nghiên cứu.](images/trangchu.png)

![Hình 4.8. Giao diện chế độ stream trong quá trình giám sát.](images/stream.png)

Một đóng góp quan trọng của đề tài là việc triển khai mô hình vào một hệ thống thực nghiệm có giao diện. Hệ thống cho phép tải ảnh, hiển thị bounding box, xem lớp và mức độ tin cậy, lưu lịch sử phân tích và quản lý theo dự án. Chế độ stream cho phép mô phỏng tình huống giám sát liên tục, nơi hệ thống không chỉ phân tích từng ảnh độc lập mà còn hỗ trợ quan sát các frame nghi ngờ theo thời gian.

Việc có một hệ thống hiển thị và lưu vết giúp tăng đáng kể giá trị nghiên cứu của đề tài. Nhiều công trình chỉ dừng ở bước công bố chỉ số mô hình mà chưa chỉ ra hệ thống sẽ được dùng ra sao. Trong nghiên cứu này, các kết quả detection không tồn tại rời rạc mà được đặt vào một luồng làm việc có ý nghĩa hơn: người dùng tải ảnh hoặc quan sát stream, hệ thống phát hiện vùng lỗi, hiển thị trực quan và lưu lại kết quả để phục vụ đối chiếu.

## 4.10. Tổng hợp đánh giá nghiệm thu

| Hạng mục | Tiêu chí | Kết quả | Đánh giá |
|---|---|---|---|
| Hiệu năng mô hình | mAP@0.5 đạt mức khá | 0.738 | Đạt |
| Điểm vận hành | Xác định được threshold rõ ràng | F1 = 0.75 tại conf 0.485 | Đạt |
| Lớp ngữ cảnh rộng | Hiệu năng cao trên lớp phức tạp | Other corruption 0.82; alligator crack 0.765 | Đạt |
| Tốc độ suy luận | Đáp ứng gần thời gian thực | ~20 ms/ảnh | Đạt |
| Tính hệ thống | Có giao diện, API, lịch sử, stream | Đã triển khai | Đạt |

Qua các tiêu chí trên, nhóm nghiên cứu đánh giá hệ thống đã đạt yêu cầu nghiệm thu ở mức khá tốt đối với một đề tài nghiên cứu ứng dụng. Kết quả không chỉ thỏa mãn khía cạnh học thuật mà còn cho thấy hệ thống có tiềm năng triển khai trong những tình huống giám sát và hỗ trợ kiểm tra thực tế.

## 4.11. Nhận xét khách quan về ưu điểm và hạn chế

Ưu điểm nổi bật nhất của hệ thống là khả năng kết hợp giữa một mô hình có cơ sở lý thuyết mạnh và một lớp triển khai ứng dụng rõ ràng. RT-DETR chứng tỏ hiệu quả trên các lớp phức tạp về ngữ cảnh, trong khi hệ thống Web/API giúp chuyển kết quả nghiên cứu thành công cụ thao tác được. Ngoài ra, việc xác định rõ threshold vận hành và phân tích sâu PR/Confusion Matrix làm tăng độ thuyết phục của báo cáo.

Hạn chế chính của đề tài nằm ở lớp transverse crack và các điều kiện dữ liệu khó. Crack mảnh, xa, thiếu tương phản hoặc bị trộn với texture nền vẫn là thách thức đáng kể. Một hạn chế khác là bounding box chưa phải biểu diễn tối ưu cho đối tượng dài, ngoằn ngoèo và nhiều nhánh. Vì vậy, dù kết quả hiện tại đã đủ tốt để khẳng định tính khả thi của hướng nghiên cứu, đề tài vẫn còn dư địa cải thiện rõ rệt trong các bước phát triển tiếp theo.
# CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1. Kết luận

Đề tài đã nghiên cứu và xây dựng thành công một hệ thống nhận diện, phân loại lỗi hư hỏng bề mặt đường bộ dựa trên kiến trúc Transformer RT-DETR với backbone ResNet50. Toàn bộ tiến trình được triển khai từ bước khảo sát lý thuyết, chuẩn hóa dữ liệu, thiết lập thực nghiệm, huấn luyện mô hình, phân tích kết quả cho đến xây dựng hệ thống nguyên mẫu và thực hiện nghiệm thu. Điều này cho phép báo cáo không chỉ trình bày kết quả mô hình mà còn mô tả đầy đủ logic nghiên cứu từ đầu đến cuối.

Về mặt định lượng, mô hình đạt mAP@0.5 tổng thể 0.738, F1 tốt nhất 0.75 tại confidence 0.485 và tốc độ suy luận khoảng 20 ms/ảnh. Các kết quả này cho thấy mô hình đạt được sự cân bằng tương đối tốt giữa độ chính xác và tốc độ, đủ để khẳng định RT-DETR là hướng tiếp cận khả thi cho bài toán hư hỏng đường bộ. Đặc biệt, hiệu năng nổi bật ở lớp other corruption và alligator crack đã cung cấp bằng chứng rõ ràng về lợi ích của Self-Attention trong việc nắm bắt ngữ cảnh toàn cục.

Về mặt ứng dụng, đề tài đã tích hợp thành công mô hình vào một hệ thống Web + API có giao diện trực quan, hỗ trợ phân tích ảnh, stream và quản lý lịch sử theo dự án. Điều này làm tăng đáng kể giá trị thực tiễn của nghiên cứu và cho thấy khả năng chuyển hóa từ mô hình học sâu sang một công cụ hỗ trợ nghiệp vụ là hoàn toàn khả thi.

## 5.2. Đánh giá thẳng thắn những hạn chế hiện tại

Mặc dù đạt kết quả khả quan, đề tài vẫn tồn tại một số hạn chế. Trước hết, transverse crack là lớp khó nhất và vẫn có mức nhầm lẫn với background khoảng 15-20%. Điều này phản ánh giới hạn của dữ liệu và cũng phản ánh hạn chế tự nhiên của biểu diễn bounding box khi phải mô tả các đối tượng rất mảnh, đứt đoạn và ít tín hiệu ngữ cảnh.

Thứ hai, dữ liệu thực địa luôn tồn tại nhiều tình huống ngoài phân bố huấn luyện như ánh sáng quá gắt, bề mặt bị loang nước, vệt sơn mờ, bóng đổ kéo dài hoặc camera chụp ở khoảng cách quá xa. Dù mô hình đã thể hiện tốt trên tập nghiệm thu hiện tại, khả năng khái quát hóa ra các điều kiện cực đoan vẫn cần được khảo sát kỹ hơn bằng các bộ dữ liệu thực địa quy mô lớn hơn.

Thứ ba, hệ thống hiện tại mới ở mức nguyên mẫu ứng dụng cục bộ. Các tính năng như phân quyền nhiều người dùng, đồng bộ dữ liệu thời gian thực, giám sát quy mô lớn hoặc tích hợp với hệ thống bản đồ/GIS vẫn chưa được triển khai trong khuôn khổ đề tài.

## 5.3. Hướng phát triển

Hướng phát triển đầu tiên mà nhóm nghiên cứu đề xuất là tối ưu mô hình bằng TensorRT hoặc ONNX Runtime để phục vụ triển khai trên các nền tảng Edge AI. Nếu giảm thêm độ trễ suy luận và tối ưu sử dụng bộ nhớ, hệ thống có thể được triển khai trên các thiết bị gắn trực tiếp lên phương tiện khảo sát hoặc camera hiện trường.

Hướng thứ hai là mở rộng dữ liệu cho các trường hợp khó, đặc biệt là transverse crack ở điều kiện ánh sáng xấu, góc chụp xiên hoặc khoảng cách lớn. Với một bài toán mà chất lượng dữ liệu đóng vai trò quyết định, việc mở rộng đúng phân bố khó có thể mang lại hiệu quả cải thiện còn lớn hơn cả việc thay đổi kiến trúc mô hình.

Hướng thứ ba là kết hợp Sensor Fusion. Trong tương lai, nhóm nghiên cứu đề xuất tích hợp thêm dữ liệu từ cảm biến quán tính, định vị, hoặc thậm chí LiDAR nếu điều kiện cho phép. Việc kết hợp nhiều nguồn tín hiệu có thể giúp hệ thống đánh giá không chỉ có lỗi hay không, mà còn hỗ trợ suy luận sâu hơn về mức độ nghiêm trọng, vị trí và tính lặp theo chuỗi thời gian.

Hướng thứ tư là chuyển từ object detection thuần bounding box sang các bài toán tinh hơn như instance segmentation hoặc crack tracing. Với các đối tượng mảnh và kéo dài, một mô hình mô tả theo vùng điểm ảnh hoặc xương (skeleton) có thể phù hợp hơn và giúp giảm nhầm lẫn ở các lớp khó.

## 5.4. Kiến nghị ứng dụng kết quả nghiên cứu

Nhóm nghiên cứu kiến nghị kết quả của đề tài có thể được ứng dụng trước hết ở vai trò công cụ hỗ trợ khảo sát hiện trường. Trong bối cảnh đó, hệ thống đóng vai trò sàng lọc tự động, gợi ý các vùng nghi ngờ và giúp người vận hành tập trung vào những vị trí cần kiểm tra kỹ hơn. Đây là cách áp dụng an toàn, thực tế và phù hợp với trạng thái hiện tại của mô hình.

Ngoài ra, hệ thống có thể được dùng trong khâu lập báo cáo hiện trạng nội bộ, quản lý dự án hoặc lưu hồ sơ kiểm tra theo thời gian. Với khả năng lưu lịch sử và trực quan hóa kết quả, công cụ này có tiềm năng hỗ trợ chuẩn hóa quy trình làm việc và nâng cao tính truy vết của dữ liệu hiện trường.

# TÀI LIỆU THAM KHẢO

[1] Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., Sekimoto, Y., & Yamamoto, T. (2022). *RDD2022: A multi-national image dataset for automatic road damage detection*. arXiv. https://doi.org/10.48550/arXiv.2209.08538

[2] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). *End-to-end object detection with transformers*. In *Computer Vision - ECCV 2020* (pp. 213-229). Springer. https://doi.org/10.1007/978-3-030-58452-8_13

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770-778). https://doi.org/10.1109/CVPR.2016.90

[4] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). *Microsoft COCO: Common objects in context*. In *Computer Vision - ECCV 2014* (pp. 740-755). Springer. https://doi.org/10.1007/978-3-319-10602-1_48

[5] Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization*. In *International Conference on Learning Representations*. https://arxiv.org/abs/1711.05101

[6] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You only look once: Unified, real-time object detection*. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 779-788). https://doi.org/10.1109/CVPR.2016.91

[7] Ultralytics. (2026). *Ultralytics documentation*. https://docs.ultralytics.com/

[8] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention is all you need*. In *Advances in Neural Information Processing Systems* (Vol. 30). https://arxiv.org/abs/1706.03762

[9] Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). *DETRs beat YOLOs on real-time object detection*. arXiv. https://doi.org/10.48550/arXiv.2304.08069

# PHỤ LỤC

## Phụ lục A. Danh mục hình minh họa đưa vào báo cáo

- Hình 4.1. Đường cong Precision-Recall của mô hình RT-DETR.
- Hình 4.2. Đường cong F1-Score theo ngưỡng confidence.
- Hình 4.3. Đường cong Precision theo ngưỡng confidence.
- Hình 4.4. Đường cong Recall theo ngưỡng confidence.
- Hình 4.5. Ma trận nhầm lẫn tuyệt đối của mô hình.
- Hình 4.6. Ma trận nhầm lẫn chuẩn hóa của mô hình.
- Hình 4.7. Giao diện tổng quan của hệ thống nghiên cứu.
- Hình 4.8. Giao diện chế độ stream trong quá trình giám sát.

## Phụ lục B. Tóm tắt kết quả nghiệm thu

| Hạng mục | Kết quả |
|---|---|
| Mô hình | RT-DETR với backbone ResNet50 |
| Số lớp | 05 lớp hư hỏng bề mặt đường bộ |
| mAP@0.5 tổng thể | 0.738 |
| F1 tốt nhất | 0.75 tại confidence 0.485 |
| Tốc độ suy luận | ~20 ms/ảnh |
| Lớp tốt nhất | Other corruption (0.82), alligator crack (0.765) |
| Lớp khó nhất | Transverse crack (0.65) |
| Nhầm với nền | Transverse crack khoảng 15-20% |

## Phụ lục C. Hình ảnh minh họa

![Giao diện tổng quan của hệ thống.](images/trangchu.png)

![Giao diện stream của hệ thống.](images/stream.png)

![PR Curve.](images/BoxPR_curve.png)

![F1 Curve.](images/BoxF1_curve.png)

![Precision Curve.](images/BoxP_curve.png)

![Recall Curve.](images/BoxR_curve.png)

![Confusion Matrix tuyệt đối.](images/confusion_matrix.png)

![Confusion Matrix chuẩn hóa.](images/confusion_matrix_normalized.png)
