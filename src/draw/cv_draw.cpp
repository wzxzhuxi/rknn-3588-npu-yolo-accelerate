
#include "cv_draw.h"

#include "utils/logging.h"

// 在img上画出检测结果
void DrawDetections(cv::Mat &img, const std::vector<Detection> &objects)
{
    NN_LOG_DEBUG("draw %ld objects", objects.size());
    for (const auto &object : objects)
    {
        cv::rectangle(img, object.box, object.color, 2);
        // class name with confidence
        std::string draw_string = object.className + " " + std::to_string(object.confidence);

        cv::putText(img, draw_string, cv::Point(object.box.x, object.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 1,
                    object.color, 2);
    }
}