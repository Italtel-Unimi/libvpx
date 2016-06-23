
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


#ifdef USE_OPENGL

#include <GLUT/glut.h>
#include "cuda/frame_cuda.h"

static GLuint texName;
GLfloat W,H;

void setup_win(void) {
	glutInit(argc_bis, argv_bis);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1000, 700);
	glutInitWindowPosition(10, 10);
	glutCreateWindow(argv_bis[0]);
}

void makeImage(int width, int height, GLubyte image[height][width][4], unsigned char *yuv_image) {
	unsigned char c;
	int k = height*width;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			c = yuv_image[k--];
			image[i][j][0] = (GLubyte) c;
			image[i][j][1] = (GLubyte) c;
			image[i][j][2] = (GLubyte) c;
			image[i][j][3] = (GLubyte) 255;
		}
}

void init(int width, int height, GLubyte (*image)[height][width][4], unsigned char *yuv_image) {
   glClearColor (1, 0.0, 0.0, 0.0);
   glShadeModel(GL_FLAT);
   glEnable(GL_DEPTH_TEST);

   makeImage(width, height, image,yuv_image);

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glGenTextures(1, &texName);
   glBindTexture(GL_TEXTURE_2D, texName);

   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width,height, 0, GL_RGBA, GL_UNSIGNED_BYTE,image);
}

void display(void) {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_TEXTURE_2D);
   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
   glBindTexture(GL_TEXTURE_2D, texName);
   glBegin(GL_QUADS);
   float w = 1.0, h = 1.0;
   glTexCoord2f(0.0, 0.0); glVertex3f(-w, -h, 0.0);
   glTexCoord2f(0.0, 1.0); glVertex3f(-w, h, 0.0);
   glTexCoord2f(1.0, 1.0); glVertex3f(w, h, 0.0);
   glTexCoord2f(1.0, 0.0); glVertex3f(w, -h, 0.0);
   glEnd();
   glFlush();
   glDisable(GL_TEXTURE_2D);
}

void reshape() {
   glViewport(0, 0, (GLsizei) W, (GLsizei) H);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(60.0, (GLfloat) W/(GLfloat) H, 1.0, 30.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glTranslatef(0.0, 0.0, -3.6);
}


void keyboard(unsigned char key) {
	switch (key) {
	case 27:     /* ESC key */
	case 'Q':
	case 'q':
		exit(0);
		break;
	case 'n':
	case 'N':

		break;
	default:
		break;
	}
}


void display_frame_GPU(unsigned char *yuv_image, int width, int height) {
	W = width;
	H = height;
	GLubyte img[height][width][4];
	GLubyte (*image)[height][width][4] = &img;

   init(width, height, image, yuv_image);
   glutDisplayFunc(display);
  // glutReshapeFunc(reshape);
   glutKeyboardFunc(keyboard);
   glutMainLoop();
   free(img);
}

#endif


#ifdef __cplusplus
}
#endif

