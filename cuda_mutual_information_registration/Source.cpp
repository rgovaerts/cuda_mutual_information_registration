
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

	int imagecut[200][200];

	Mat image1(Size(wdt, hgt), CV_8UC3);
	
	int colCheck;
	int rowCheck;
	int templateCOL = 50;
	int templateROW = 50;
	colCheck = wdt%templateCOL;
	rowCheck = hgt%templateROW;
	
	// A compl�ter encore ( rajouter 2 if ainsi que completer le premier pour qu'il "d�coupe" bien en carr� et pas en ligne (m�me si on peut le stoquer sous forme de vecteur dans un tableau )
	if (colCheck == 0 && rowCheck == 0) // sert a check si la taille choisie pour l'image d�coup�e permet de recouvrir l'image compl�te de mani�re simple
	{ 
		for (int i = 0; i < wdt; i++)
		{
			for (int j = 0; j < hgt; j++)
			{
						grayscale.row(j).col(i).copyTo(image1.row(j).col(i));  
			}
		}
	}



	return 0;
}*/