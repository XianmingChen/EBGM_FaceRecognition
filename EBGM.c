#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include "EBGM_FeatureVectors.h"
#include "EBGM_FaceComparison.h"
double Gabor_Respone[Filter_Num][Height][Width][2];
double Feature_Vectors[Total_train_face][500][41][2];

void read_image(char *filepath,double image[][Width])
{
	int i,j;
	CvScalar s;
	IplImage *img=cvLoadImage(filepath,0);
    for(i=0;i<img->height;i++)
	{
        for(j=0;j<img->width;j++)
		{
			s=cvGet2D(img,i,j); 
			image[i][j]=s.val[0]/255;
        }
    }
    cvReleaseImage(&img);  
}


double ****Malloc4D(double ****Array,int n1,int n2,int n3,int n4)
{
	int i,j,k,m;
	Array=(double ****)malloc(n1*sizeof(double ***));
	for (i=0;i<n1;i++)
	{
		Array[i]=(double ***)malloc(n2*sizeof(double **));
		for (j=0;j<n2;j++)
		{
			Array[i][j]=(double **)malloc(n3*sizeof(double *));
			for (k=0;k<n3;k++)
			{
				Array[i][j][k]=(double *)malloc(n4*sizeof(double));
				for (m=0;m<n4;m++)
				{
					Array[i][j][k][m]=0.0;
				}
			}
		}
	}
	return Array;
}

double ****Free4DArray(double ****Array,int n1,int n2,int n3)
{
	int i,j,k;
	for (i=0;i<n1;i++)
	{
		for (j=0;j<n2;j++)
		{
			for (k=0;k<n3;k++)
			{
				free(Array[i][j][k]);
				Array[i][j][k]=0;
			}
		}
	}		
	for (i=0;i<n1;i++)
	{
		for (j=0;j<n2;j++)
		{
			free(Array[i][k]);
			Array[i][k]=0;
		}
	}
	for (i=0;i<n1;i++)
	{
		free(Array[i]);
		Array[i]=0;
	}
	free(Array);
	Array=NULL;
	return NULL;
}


void main()
{
	char image_path[255];
	int i;
	double trainface[Height][Width]={0.0};
	//double Gabor_Respone[Filter_Num][Height][Width][2]={0.0};

	double Mean_Value[Filter_Num][2]={0.0};
	double Each_Feature_Vectors[500][41][2]={0.0};
	//double Feature_Vectors[Total_train_face][500][41][2]={0.0};

	int train_feature_count[Total_train_face]={0};
	int each_feature_count=0;
	int probe_feature_count=0;

	int Accuracy=0;
	int Probe_count=0;
	double start,end;
	double total_start,total_end;
	FILE *record;

	record=fopen("result_EBGM.txt","a+");
	total_start=clock();
	start=clock();
	printf("%d images of regular EBGM begins...\n",Total_probe_face);
	for (i=0;i<Total_train_face;i++)
	{
		printf("Training image %d...\n",i+1);
		sprintf(image_path,"Aligned_FERET/input/trainfaces/%d.jpg",i+1);
		read_image(image_path,trainface);
		GaborFilterResponse(trainface,Mean_Value);
		EBGM_FeatureVectors(Mean_Value,&each_feature_count,Each_Feature_Vectors);
		train_feature_count[i]=each_feature_count;
		each_feature_count=0;
		memcpy(Feature_Vectors[i],Each_Feature_Vectors,500*41*2*8);
	}
	end=clock();
	start=end-start;
	fprintf(record,"The time for %d images traing is: %f ms.\n",Total_train_face,start);

	printf("Begin to probe images...\n");
	start=clock();
	for (i=0;i<Total_probe_face;i++)
	{
		printf("Image %d Probing...\n",i+1);
		sprintf(image_path,"Aligned_FERET/input/probefaces/%d.jpg",i+1);
		read_image(image_path,trainface);
		GaborFilterResponse(trainface,Mean_Value);
		EBGM_FeatureVectors(Mean_Value,&each_feature_count,Each_Feature_Vectors);
		probe_feature_count=each_feature_count;
		each_feature_count=0;
		Accuracy=EBGM_FaceComparison(Total_train_face,train_feature_count,probe_feature_count,Each_Feature_Vectors,i);

		if (Accuracy==1)
		{
			Probe_count++;
			printf("Image %d Probe successfully!\n",i+1);
		}
		else
		{
			printf("Image %d Probe failed!\n",i+1);
		}
	}
	end=clock();
	start=end-start;
	fprintf(record,"The time for %d images probing is: %f ms.\n",Total_probe_face,start);

	printf("%d Images probe completed!\n",Total_probe_face);
	total_end=clock();
	total_start=total_end-total_start;
	printf("The total running time is %f ms.\n",total_start);
	fprintf(record,"The total running time is %f ms.\n",total_start);
	printf("Probe accuary of %d images in single solution is:= %f\n",Total_probe_face,(double)Probe_count/Total_probe_face);
	fprintf(record,"Probe accuary of %d images in single solution is:= %f\n",Total_probe_face,(double)Probe_count/Total_probe_face);
	fprintf(record,"\n");
	fclose(record);

	system("pause");
}