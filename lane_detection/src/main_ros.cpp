#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <numeric>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "LaneDetection.h"
#include <ros/ros.h>						 //ros 的头文件

#include <image_transport/image_transport.h> //image_transport			    
#include <cv_bridge/cv_bridge.h>	     //cv_bridge
#include <sensor_msgs/image_encodings.h>	 //图像编码格式，包含对图像进行编码的函数。
#include <opencv2/imgproc/imgproc.hpp>		 //图像处理
#include <opencv2/highgui/highgui.hpp>		 //opencv GUI


//---msg的头，需要发布的信号---------
#include "lane_detection/ld_Frame.h"
#include "lane_detection/ld_Point.h"
#include "lane_detection/ld_Coeff.h"
#include "lane_detection/ld_LaneParam.h"

#include "JudgeLane.h"
#include <chrono>

std::vector<float> hist_time;


CJudgeCenter JudgeCenter ;
static const std::string OPENCV_WINDOW = "Image window"; //申明一个GUI 的显示的字符串
LaneDetection lane;
double pixel_ratio=40; //world -->> pixel
double thres_ldw=0.65; //warning threshold

int predict_num=0;  //连续预测次数
bool predict_continue_flag = true;  //连续预测flag
std::vector<float> hist_v_LeftDis;
std::vector<float> hist_v_RightDis;
std::vector<float> hist_v_LaneWidth;
std::vector<float> hist_v_RadiusOfCurve;


// Initializing variables depending on the resolution of the input image.
// 根据车辆坐标系的方程，返回方程值(真实尺寸)
double valueAtIPM(std::vector<float>& f, float x) {
	float ans = 0.f;
	for (int i = (int)f.size() - 1; i >= 0; --i)
		ans = ans * x + f[i];
	return ans;
}
std::vector<LD_COEFF> CurveByParallel(std::vector<LD_COEFF> _ld_coeff)
{
	std::vector<LD_COEFF> tmp_ld_coeff;
}

void vector_InitValue(std::vector<float> & vector_a,float value,int num)
{
	if(vector_a.size()!=0){
		vector_a.clear();
	}else{
		for(int ii=0;ii<num;ii++){
		vector_a.push_back(value);
		}
	}
}

void vector_Update(std::vector<float> & vector_a,float value)
{
	vector_a.push_back(value);
	vector_a.erase(vector_a.begin());
}
class ImageConverter //申明一个图像转换的类
{
	ros::NodeHandle nh_; //实例化一个节点
	ros::NodeHandle nh_param;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_; //订阅节点
	image_transport::Publisher image_pub_;  //发布节点

	ros::Publisher pub_frame;//publisher frame(lane pixel/lane world)
	ros::Publisher image_mat_pub;			//发布节点
	std::string strCameraSub;
	std::string strCameraRosPub;
	std::string strCameraId;

	bool verbose_lm_detction;
	bool verbose_seed_gen;
	bool verbose_run_crf;
	bool verbose_validating;
	bool verbose_kalman ;
	bool verbose;
	

	
	//msg obj
	lane_detection::ld_Frame ld_obj;
   public:
	ImageConverter():it_(nh_),nh_param("~")
	{
	verbose_lm_detction = false;
	verbose_seed_gen = false;
	verbose_run_crf = false;
	verbose_validating = false;
	verbose_kalman=false;
	verbose = verbose_lm_detction | verbose_seed_gen | verbose_run_crf | verbose_validating;

	nh_param.param<std::string>("CameraSub", strCameraSub, "/camera1/image_raw"); //skou
	nh_param.param<std::string>("CameraRosPub", strCameraRosPub, "/lane_detection/output_video");
	nh_param.param<std::string>("CameraId", strCameraId, "");

	image_sub_ = it_.subscribe(strCameraSub, 1, &ImageConverter::imageCb, this);
	image_pub_ = it_.advertise(strCameraRosPub, 1);
	pub_frame = nh_.advertise<lane_detection::ld_Frame>("lane_data", 1);

	}

	~ImageConverter()
	{
	//	cv::destroyWindow(OPENCV_WINDOW);
	}

	void imageCb(const sensor_msgs::ImageConstPtr &msg) //回调函数
	{
		double resize_time1 = clock();
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		
		//将ros中的图像类型由sensor image 转为 Mat 
		cv_bridge::CvImagePtr cv_ptr; //申明一个CvImagePtr
		try
		{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		}
		catch (cv_bridge::Exception &e)
		{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
		}
		
		
		// Update GUI Window
		cv::Mat img_resize1;
		cv::Mat img_resize;
		cv::Mat img_result;

		std::cout << "temp safe..." << std::endl;
		cv::resize(cv_ptr->image, img_resize1, cv::Size(768, 480)); //可设置外部接口，待改进
		cv::Rect rect(128,150,512,256);
		img_resize = img_resize1(rect);
		std::cout << "temp safe..." << std::endl;

		double resize_time2 = clock();
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
		std::cout<<"resize and crop time: "<< (resize_time2 - resize_time1) <<" mms" <<std::endl ;
		std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		std::cout << " time cost= " << (time_used.count() * 1000) << " ms." << std::endl;


		process(img_resize, img_result);

		// cv::imshow("Lane Detection",img_result);

		cv_ptr->image = img_result.clone();
		cv::waitKey(2);
		image_pub_.publish(cv_ptr->toImageMsg());
	}
	int process(cv::Mat srcImg, cv::Mat &procImg)
	{

	
		if (!lane.initialize_variable(srcImg))
		{
			return -1;
		}

		double main_time1 = clock();
		std::cout << "......................." << std::endl;
		std::cout << "new" << std::endl;
		if (!lane.initialize_Img(srcImg))
		{
			return 0;
		}

		double process_time1 = clock();
		// detecting lane markings
		lane.lane_marking_detection(verbose_lm_detction);
		double process_time2 = clock();
		
		// supermarking generation and low-level association
		lane.seed_generation(verbose_seed_gen);
		double process_time3 = clock();

		// CRF graph configuration & optimization using hungarian method
		lane.graph_generation(verbose_run_crf);
		double process_time4 = clock();
	
		lane.validating_final_seeds(verbose_validating);
		double process_time5 = clock();

		std::cout << "process_lane_marking " << (process_time2 - process_time1) << "mms" << std::endl;
		std::cout << "process_seed_generation " << (process_time3 - process_time2) << "mms" << std::endl;
		std::cout << "process_graph_generation " << (process_time4 - process_time3) << "mms" << std::endl;
		std::cout << "process_final_seed " << (process_time5 - process_time4) << "mms" << std::endl;

		
		procImg = lane.kalman(verbose_kalman).clone();
		double process_time6 = clock();

		std::cout << "process_kalman" << (process_time6 - process_time5) << "mms" << std::endl;


		lane_detection::ld_Point tmp_Point;
		lane_detection::ld_Coeff tmp_Coeff;
		lane_detection::ld_LaneParam tmp_LaneParam_Pixel;
		lane_detection::ld_LaneParam tmp_LaneParam_World;

		ld_obj.lane_Pixel.clear();
		ld_obj.lane_World.clear();
		ld_obj.lane_Coeff.clear();

		std::vector<double> candidate_dis; 
		//候选ID
		std::vector<int> candidate_id; 

		candidate_id.clear();
		candidate_dis.clear();

		float lane_angle = 0.0;
		float dist_bias_1 = 0.0; //the distace in 1 meter
		float dist_bias_5 = 0.0; //the distace in 5 meter

		std::vector<float> Lane_angle(2);
		std::vector<float> Dist1(2);
		std::vector<float> Dist5(2);
		Lane_angle.clear();
		Dist1.clear();
		Dist5.clear();

		for (int i = 0; i < lane.v_PolyfitParam.v_ld_coeff.size(); i++)
		{

			tmp_Coeff.a=lane.v_PolyfitParam.v_ld_coeff[i].a;
			tmp_Coeff.b=lane.v_PolyfitParam.v_ld_coeff[i].b;
			tmp_Coeff.c=lane.v_PolyfitParam.v_ld_coeff[i].c;
			tmp_Coeff.d=lane.v_PolyfitParam.v_ld_coeff[i].d;
			tmp_Coeff.id=lane.v_PolyfitParam.v_ld_coeff[i].global_id;
			ld_obj.lane_Coeff.push_back(tmp_Coeff);

			cv::Point2f dot_p;
			std::vector<float> coeff(4);
			coeff[3] = tmp_Coeff.a;
			coeff[2] = tmp_Coeff.b;
			coeff[1] = tmp_Coeff.c;
			coeff[0] = tmp_Coeff.d;

			
			dot_p.x = 0;
			dot_p.y =valueAtIPM(coeff, 0);
			if (-3.5<dot_p.y && dot_p.y<3.5){								
				lane_angle = atan(3*coeff[3]*dot_p.x*dot_p.x+2*coeff[2]*dot_p.x+coeff[1]);					
				candidate_dis.push_back(dot_p.y);
				candidate_id.push_back(i);
				Lane_angle.push_back(lane_angle);				
			
			}
			
		}

		double left_0 = 0.0;
		double right_0 = 0.0;
		int flag_left=0;
		int flag_right=0;
		

		bool b_flag1=true;
		bool b_flag2=true;

		int i_flag_l=-1;
		int i_flag_r=-1;

		if (candidate_dis.size()>1){
			for(int ii=0;ii<candidate_dis.size();ii++){
				if(candidate_dis[ii]<0){
					if(b_flag1){
					left_0 = candidate_dis[ii];
					i_flag_l=ii;
					}
					b_flag1=false;

					if(candidate_dis[ii]>left_0){
					left_0 =candidate_dis[ii];
					i_flag_l=ii;
					}
					flag_left=1;
				}
				if(candidate_dis[ii]>0){
					if(b_flag2){
					right_0 = candidate_dis[ii];
					i_flag_r=ii;
					}
					b_flag2=false;

					if(candidate_dis[ii]<right_0){
					right_0 =candidate_dis[ii];
					i_flag_r=ii;
					}	
					flag_right=1;
				}
			}
		}
		else
		{
			ROS_INFO("lane detection is not reliable OR no lane ");
		}
		
		double angle_final =0.0;
		double curve_radius = 0.0;
		float curve_position_x= 0.0;
		if(i_flag_l==-1 && i_flag_r!=-1){
			angle_final = Lane_angle[i_flag_r];
			curve_radius = fabs(pow(1.0+pow((ld_obj.lane_Coeff[candidate_id[i_flag_r]].c),2),(3/2))/(2*ld_obj.lane_Coeff[candidate_id[i_flag_r]].b));

		}else if(i_flag_l!=-1 && i_flag_r==-1) {
			angle_final = Lane_angle[i_flag_l];
			curve_radius = fabs(pow(1.0+pow((ld_obj.lane_Coeff[candidate_id[i_flag_l]].c),2),(3/2))/(2*ld_obj.lane_Coeff[candidate_id[i_flag_l]].b));

		}else if(i_flag_l!=-1 && i_flag_r!=-1) {
			angle_final = (Lane_angle[i_flag_l]+Lane_angle[i_flag_r])/2;
			curve_radius = (fabs(pow(1.0+pow((ld_obj.lane_Coeff[candidate_id[i_flag_l]].c),2),(3/2))/(2*ld_obj.lane_Coeff[candidate_id[i_flag_l]].b))+fabs(pow(1.0+pow((ld_obj.lane_Coeff[candidate_id[i_flag_r]].c),2),(3/2))/(2*ld_obj.lane_Coeff[candidate_id[i_flag_r]].b)))/2.0;
		}
		if(curve_radius>3000){curve_radius=3000;}
		double mid = 0;
		double bias_dis = 0;
		double lane_width = 3.5;
		char tmp_str[20]="";

		
		if(flag_left && flag_right && (fabs(left_0)+fabs(right_0))* cos(angle_final)> 3.0)
		{
			mid =(left_0+right_0)/2;
			bias_dis = mid * cos(angle_final);
			lane_width = (fabs(left_0)+fabs(right_0))* cos(angle_final);
			//如果道路宽度有突变，则参考历史容器中的大小，做为纠正值
			if (fabs(lane_width-hist_v_LaneWidth[hist_v_LaneWidth.size()-1])>0.3)
			{
				for(int ii=0;ii<5;ii++){
				lane_width += hist_v_LaneWidth[hist_v_LaneWidth.size()-1-ii];
				}
				lane_width = lane_width/6.0;
			}
			//将道路宽度加入历史容器中
			vector_Update(hist_v_LaneWidth,lane_width);
			//连续预测标志为false
			predict_continue_flag = false;	
			
		}else{
			for(int ii=0;ii<10;ii++){
			lane_width += hist_v_LaneWidth[hist_v_LaneWidth.size()-1-ii];
			}
			lane_width = lane_width/11.0;
			vector_Update(hist_v_LaneWidth,lane_width);
			predict_num++;
			predict_continue_flag =true;
		}
		if(!predict_continue_flag){
			predict_num = 0;
		}
		//连续预测次数超过五次，还未有真实检测值的时候，报警
		if(predict_continue_flag && predict_num>5)
		{//一些补尝措施
			
		}
		//根据道路宽进行车道线方程去伪。
		double judge_time1 = clock();
		JudgeCenter.SetParam(lane_width,lane_width*0.1,angle_final);
		JudgeCenter.Run(lane.v_PolyfitParam.v_ld_coeff);
		double judge_time2 = clock();
		std::cout << "judge_time " << (judge_time2 - judge_time1) << "mms" << std::endl;
		ld_obj.lane_Coeff.clear();
		for (int i = 0; i < lane.v_PolyfitParam.v_ld_coeff.size(); i++)
		{	
			tmp_Coeff.a=lane.v_PolyfitParam.v_ld_coeff[i].a;
			tmp_Coeff.b=lane.v_PolyfitParam.v_ld_coeff[i].b;
			tmp_Coeff.c=lane.v_PolyfitParam.v_ld_coeff[i].c;
			tmp_Coeff.d=lane.v_PolyfitParam.v_ld_coeff[i].d;
			tmp_Coeff.id=lane.v_PolyfitParam.v_ld_coeff[i].global_id;
			ld_obj.lane_Coeff.push_back(tmp_Coeff);

		}

		ld_obj.header.stamp=ros::Time::now();
		ld_obj.lane_width =lane_width ;
		ld_obj.bias_dis = bias_dis;
		ld_obj.cl_flag = flag_left && flag_right ;
		ld_obj.bias_theta = angle_final;
		ld_obj.curve_radius =curve_radius;

		pub_frame.publish(ld_obj);

		cv::waitKey(1);

		double main_time2 = clock();
		std::cout << "main_time " << (main_time2 - main_time1) << "mms" << std::endl;
		// std::cout << "FPS: " << 1000000 / (main_time2 - main_time1) << std::endl;
		hist_time.push_back(main_time2 - main_time1);
		double sum_hist_time = std::accumulate(std::begin(hist_time), std::end(hist_time), 0.0);
		double mean_hist_time = sum_hist_time / hist_time.size(); //均值
		std::cout << "mean_hist_time " << mean_hist_time << "mms" << std::endl;
		// std::cout << "mean_FPS: " << 1000000 / mean_hist_time << std::endl;
		return 0;
	}
};




int main(int argc, char **argv)
{ //
	ros::init(argc, argv, "lane_detection", ros::init_options::AnonymousName);
	hist_v_LeftDis.clear();
	hist_v_RightDis.clear();
	hist_v_LaneWidth.clear();
	hist_v_RadiusOfCurve.clear();
	vector_InitValue(hist_v_LaneWidth,3.5,10);
	ImageConverter ic;  //类ImageConverter、对象ic
	ros::spin();  //ROS消息回调处理函数
	return 0;
}


