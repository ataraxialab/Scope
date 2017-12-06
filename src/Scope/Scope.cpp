//
// Created by 赵之健 on 2017/12/5.
//

#include <tron_algorithm.hpp>
#include "../common/log.hpp"
#include "../common/util.hpp"
#include "atlab_scope.h"


#define USE_KCF
#if defined(USE_KCF)
#include "../ot/kcf.hpp"
#endif

#define MAX_OBJECT_NUM 15

#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>


struct ScopeEngine {
  TronEngine ssd_;
  VecBoxF boxes_;
  VecInt uids_;

  mongocxx::client conn;
#if defined(USE_KCF)
  OT *ot_;
  int od_count_ = -1, od_period_ = 2;
#endif
};

inline VecBoxF remove_box(VecBoxF boxes) {
  for (int i = 0; i < boxes.size(); ++i) {
    BoxF &box_i = boxes[i];
    if (box_i.label != 1) continue;
    for (int j = i + 1; j < boxes.size(); ++j) {
      BoxF &box_j = boxes[j];
      if (box_j.label != 1) continue;
      float height_i = box_i.ymax - box_i.ymin;
      float height_j = box_j.ymax - box_j.ymin;
      float width_i = box_i.xmax - box_i.xmin;
      float width_j = box_j.xmax - box_j.xmin;
      float dist = (box_i.xmax + box_i.xmin - box_j.xmax - box_j.xmin) / 2;
      if (dist < (width_i + width_j) / 2 + 50) {
        float delta_top = std::abs(box_i.ymin - box_j.ymin);
        float delta_bottom = std::abs(box_i.ymax - box_j.ymax);
        if (delta_top < 20 || delta_bottom < 20) {
          float size_i = Boxes::Size(box_i), size_j = Boxes::Size(box_j);
          if (size_i > size_j && (height_j / width_j) > 1.5) {
            box_j.label = -1;
          }
          if (size_i < size_j && (height_i / width_i) > 1.5) {
            box_i.label = -1;
          }
        }
      }
    }
  }
  VecBoxF out_boxes;
  for (const auto &box : boxes) {
    if (box.label != -1) out_boxes.push_back(box);
  }
  boxes.clear();
  return out_boxes;
}

SCOPE_API MRESULT Scope_Initial(SCOPE_ENGINE *scope_engine) {
  if (Timer::is_expired(0, 1, 0)) {
    LOG(FATAL) << "This program is expired, compiled on "
               << Timer::get_compile_time_str() << ".";
  }

  ScopeEngine *vpd_engine_ = new ScopeEngine();

  TronInput tron_input = {};
  tron_input.gpu_id = 0;
  std::string model_ssd =
      "models/adas_model_finetune_reduce_3_merged.shadowmodel";
  tron_input.model = const_cast<char *>(model_ssd.c_str());
  int status = TronInitial(tron_input, &vpd_engine_->ssd_);

#if defined(USE_KCF)
  vpd_engine_->ot_ = new OT(MAX_OBJECT_NUM);
#endif

  // get mongodb
  mongocxx::instance inst{};
  mongocxx::uri uri("mongodb://localhost:27017");
  vpd_engine_->conn = mongocxx::client(uri);

  *scope_engine = vpd_engine_;

  return 0;
}

#if defined(USE_OpenCV)
SCOPE_API MRESULT Scope_Detect(SCOPE_ENGINE scope_engine, const cv::Mat &img_data,
                           PTR_SCOPE_OUTPUT scope_output) {
  ScopeEngine *ptr = reinterpret_cast<ScopeEngine *>(scope_engine);

#if defined(USE_KCF)
  ptr->od_count_ = ++ptr->od_count_ % ptr->od_period_;
  if (!ptr->od_count_) {
    std::vector<VecBoxF> Bboxes_;
    TronDetectionOutput tron_detection_output = {};
    tron_detection_output.num_objects = 0;

    imwrite("test.jpg",img_data);
    int status =
        TronDetection(ptr->ssd_, "test.jpg", &tron_detection_output);
//    ptr->ssd_->Predict(img_data, rois, &Bboxes_);

    VecBoxF Bbox;
    for(int i =0;i<tron_detection_output.num_objects;i++){
      BoxF single_box;
      single_box.label = tron_detection_output.objects[i].id;
      single_box.score = tron_detection_output.objects[i].score;
      single_box.xmin = tron_detection_output.objects[i].xmin;
      single_box.xmax = tron_detection_output.objects[i].xmax;
      single_box.ymin = tron_detection_output.objects[i].ymin;
      single_box.ymax = tron_detection_output.objects[i].ymax;
      Bbox.push_back(single_box);
    }
    Bboxes_.push_back(Bbox);

    ptr->ot_->FeedOD(img_data, remove_box(Boxes::NMS(Bboxes_, 0.5)));
  }
  ptr->ot_->Tracking(img_data, &ptr->boxes_,&ptr->uids_);

#else
  std::vector<VecBoxF> Bboxes_;
  ptr->ssd_->Predict(img_data, rois, &Bboxes_);
  Boxes::Amend(&Bboxes_, rois);
  ptr->boxes_ = remove_box(Boxes::NMS(Bboxes_, 0.5));
#endif

  //std::cout<< "tron_detection_output" <<std::endl;
  LOG(INFO)<<"tron_detection_output";
  scope_output->num_objects =
      std::min(static_cast<int>(ptr->boxes_.size()), MAX_OBJECT_NUM);

  for (int i = 0; i < scope_output->num_objects; ++i) {
    const BoxF &box = ptr->boxes_[i];
    scope_output->objects[i].left = (MInt32)box.xmin;
    scope_output->objects[i].top = (MInt32)box.ymin;
    scope_output->objects[i].right = (MInt32)(box.xmax);
    scope_output->objects[i].bottom = (MInt32)(box.ymax);
    scope_output->idx[i] = (MInt32)(ptr->uids_[i]);
    scope_output->labels[i] = box.label;
  }
  LOG(INFO)<< "num of boxes = "<<scope_output->num_objects;
  //std::cout<<"num of boxes = "<<scope_output->num_objects<<std::endl;

  return 0;
}

#endif

SCOPE_API MRESULT Scope_WriteDB(SCOPE_ENGINE scope_engine,
                                std::vector<SAVE_INFO> save_infos){
  ScopeEngine *ptr = reinterpret_cast<ScopeEngine *>(scope_engine);
  auto collection = ptr->conn["scope"]["test"];

  auto builder = bsoncxx::builder::stream::document{};
  std::vector<bsoncxx::document::value> documents;

  for(std::vector<SAVE_INFO>::iterator iterator = save_infos.begin();
      iterator!=save_infos.end();iterator++){

      bsoncxx::document::value doc_value = builder
                << "idx" << iterator->idx
                << "filename" << iterator->FileName
                << "frameidx" << iterator->FrameIdx
                << "bbox" << bsoncxx::builder::stream::open_array
                    <<bsoncxx::builder::stream::open_document
                    << "x" << iterator->BBox.left
                    << "y" << iterator->BBox.top
                    << bsoncxx::builder::stream::close_document
                    <<bsoncxx::builder::stream::open_document
                    << "x" << iterator->BBox.right
                    << "y" << iterator->BBox.top
                    << bsoncxx::builder::stream::close_document
                    <<bsoncxx::builder::stream::open_document
                    << "x" << iterator->BBox.right
                    << "y" << iterator->BBox.bottom
                    << bsoncxx::builder::stream::close_document
                    <<bsoncxx::builder::stream::open_document
                    << "x" << iterator->BBox.left
                    << "y" << iterator->BBox.bottom
                    << bsoncxx::builder::stream::close_document
                << bsoncxx::builder::stream::close_array
                << "score" << iterator->score
                << "label" << iterator->label
                << bsoncxx::builder::stream::finalize;

      documents.push_back(doc_value);
  }
  if (documents.size() >0) collection.insert_many(documents);



  return 0;
}


SCOPE_API MRESULT Scope_Release(SCOPE_ENGINE scope_engine) {
  ScopeEngine *ptr = reinterpret_cast<ScopeEngine *>(scope_engine);

  if (ptr->ssd_ != nullptr) {
    TronRelease(ptr->ssd_);
    ptr->ssd_ = nullptr;
  }

  ptr->boxes_.clear();

#if defined(USE_KCF)
  if (ptr->ot_ != nullptr) {
    delete ptr->ot_;
    ptr->ot_ = nullptr;
  }
#endif

  if (ptr != nullptr) {
    delete ptr;
    ptr = nullptr;
  }

  return 0;
}
