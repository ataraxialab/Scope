//
// Created by 赵之健 on 2017/12/5.
//

#include "../Scope/atlab_scope.h"
#include "sstream"

using namespace cv;
int main() {
  SCOPE_ENGINE vpd_engine;

  MRESULT mresult = Scope_Initial(&vpd_engine);
  String filename  = "mp4.mp4";

  if (mresult != 0){
    std::cout<<"init error"<<std::endl;
  }


  // Set up tracker.
  // Instead of MIL,you can also use
  // BOOSTING, KCF,TLD, MEDIANFLOW or GOTURN
  // Ptr<Tracker> tracker= Tracker::create( "MIL");

  // Read video
  VideoCapture video(filename.c_str());
  Size s = Size((int) video.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
                (int) video.get(CV_CAP_PROP_FRAME_HEIGHT));
  int ex = static_cast<int>(video.get(CV_CAP_PROP_FOURCC));

  VideoWriter  writer("result_1.mp4",ex, video.get(CV_CAP_PROP_FPS),s);

  // Check video isopen
  if(!video.isOpened())
  {
    std::cout<< "Could not read video file"<< std::endl;
    return 1;
  }

  // Read firstframe.
  Mat frame;
  video.read(frame);

  // Define aninitial bounding box
  // Rect2d bbox(287,23, 86, 320);

  // Uncomment theline below if you
  // want to choosethe bounding box
  // bbox =selectROI(frame, false);

  // Initializetracker with first frame and bounding box
  //tracker->init(frame,bbox);

  //PTR_VPD_OUTPUT output;
  SCOPE_OUTPUT vpd_output ;
  vpd_output.objects = new MRECT[20];
  vpd_output.labels = new MInt32[20];
  vpd_output.idx = new MInt32[20];
  vpd_output.scores = new float[20];

  std::vector<SAVE_INFO> infos;
  int frameidx =0;

  while(video.read(frame))
  {
    //Update tracking results
    //tracker->update(frame,bbox);

    Rect2d bbox;

    Scope_Detect(vpd_engine, frame, &vpd_output);
    std::cout<<"detect finished"<<std::endl;
    for (int i = 0;i<vpd_output.num_objects;i++){
      //Draw bounding box
      bbox.x = vpd_output.objects[i].left;
      bbox.y = vpd_output.objects[i].top;
      bbox.width = vpd_output.objects[i].right -
          vpd_output.objects[i].left;
      bbox.height = vpd_output.objects[i].bottom -
          vpd_output.objects[i].top;

      std::ostringstream ss;
      ss << vpd_output.idx[i];



      std::string text;
      text =  ss.str();
      int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
      double fontScale = 1;
      int thickness = 2;
      cv::Point textOrg(bbox.x, bbox.y-10);
      cv::putText(frame, text, textOrg, fontFace, fontScale, Scalar::all(0), thickness,8);

      if (vpd_output.labels[i] == 1){
    //  int color_r = vpd_output.idx[i] * 1000 %255;
   //   int color_g = vpd_output.idx[i] * 500 %255;
   //   int color_b = vpd_output.idx[i] * 700 %255;
      rectangle(frame,bbox, Scalar( 255, 0, 0 ), 2, 1 );
      } else{
        rectangle(frame,bbox, Scalar( 0, 255, 0 ), 2, 1 );
      }

    }

    //Display result
    imshow("Tracking",frame);
    writer.write(frame);
    for (int i = 0; i < vpd_output.num_objects; ++i) {
        SAVE_INFO temp;
        temp.score = vpd_output.scores[i];
        temp.label = vpd_output.labels[i];
        temp.BBox  = vpd_output.objects[i];
        temp.FileName = filename.c_str();
        temp.FrameIdx = frameidx;
        temp.idx = vpd_output.idx[i];
        infos.push_back(temp);
    }
    Scope_WriteDB(vpd_engine,infos);
    int k = waitKey(10);
    if(k== 27) break;
    frameidx++;

  }
  video.release();
  writer.release();
  Scope_Release(vpd_engine);
  delete []vpd_output.objects;
  delete []vpd_output.labels;
  delete []vpd_output.scores;
  delete []vpd_output.idx;


  return 0;



}
