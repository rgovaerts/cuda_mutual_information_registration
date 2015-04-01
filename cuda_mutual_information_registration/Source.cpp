
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include "opencv\cv.h"

using namespace cv;






/*int main()
{
	Mat image = imread("Lolol.png", CV_LOAD_IMAGE_COLOR);
	Mat grayscale;
	cvtColor(image, grayscale, CV_BGR2GRAY);


	int hgt = grayscale.rowRange;
	int wdt = grayscale.colRange;


	//Mat image1(Size(wdt, hgt), CV_8UC3);

	

	int colCheck;
	int rowCheck;
	int templateCOL = 50;
	int templateROW = 50;
	colCheck = wdt%templateCOL;
	rowCheck = hgt%templateROW;
	

	

	// A compléter encore ( rajouter 2 if ainsi que completer le premier pour qu'il "découpe" bien en carré et pas en ligne (même si on peut le stoquer sous forme de vecteur dans un tableau )
	if (colCheck == 0 && rowCheck == 0) // sert a check si la taille choisie pour l'image découpée permet de recouvrir l'image complète de manière simple
	{ 
		int imageCUT [((hgt/templateROW)*(wdt/templateCOL))-1][(templateCOL*templateROW)-1]; //ce sera plus facile de filer un pointeur au kernel qu'un tableau, non ?
		int index3=0;
		int index4=0;
		for (int index=0; index < wdt/templateCOL;index++)
		{
			for (int index2=0; index2 <hgt/templateROW;index2++)
			{
			
				for (int i = templateCOL*index; i < templateCOL+templateCOL*index; i++)
				{
					
					for (int j = templateROW*index2; j < templateROW+templateROW*index2; j++)
					{
							grayscale.row(j).col(i).copyTo(imageCUT[index3][index4]);
							
							index4++;
					}
					
					
				}
				index3++;
				index4=0;
			}
		}
	}

	return 0;
}*/