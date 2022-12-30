#include <iostream>
#include <random>
#include <vector>
#include <fstream>

int N_for_an = pow(2, 10);


int N =  pow(2, 10);
double left = 0;
double right = 15;
double dt = (right-left) / N;
double Dt = (right - left) / N_for_an;
double x0 = 1;
double m = 0;
double k = 2;

double a = -1;
double b = 1;


double x01 =4;
double x02 = 2;
double alpha =3;

double error = 0;
double sr1 =0;
double sr2 = 0;
double xan = 0;
double xt = 0;

double A1(double x1, double x2)
{
	return x1*(1-x2);
//	return -(a + b * b * x1) * (1 - x1 * x1);
	//return  k * x1;
}

double A2(double x1, double x2)
{
	//return 0;
	return alpha*x2*(x1-1);
}
double B1(double x1, double x2)
{
	return  0.5*x1 ;
	//return b * (1 - x1 * x1);;
	//return m * x1;
}
double B2(double x1, double x2)
{
	return 0.5 * x2;
	//return 1;
}



double A( double t)
{
	return k * t;
	//return -(a + b * b * t) * (1 - t * t);
	///return t * t + t;
}
double B(double t)
{
	return m * t;
	//return  b * (1 - t * t);
}


double foo(double gauss_val, double last_val )
{
	//return last_val +  exp((k - 0.5 * m * m) * x + m* val);   
	return last_val + k * last_val * dt+ m * last_val* gauss_val;
	//return last_val  -(a + b * b * last_val) * (1 - last_val * last_val)*dt+ b * (1 - last_val * last_val) * gauss_val;
//return last_val + 0.25 * last_val*dt + sqrt(last_val) * gauss_val;
}
	
double foo2(double t ,double gauss_val)
{
	//return last_val +  exp((k - 0.5 * m * m) * x + m* val); 
	return x0*exp((k-m*m/2)*t + m * gauss_val);
	//return ((1 + x0)*exp(-2 * a * t + 2 * b * gauss_val) + x0 - 1) /
		//((1 + x0)*exp(-2 * a * t + 2 * b * gauss_val) - x0 + 1);
	//return (1 + 0.5 * gauss_val) * (1 + 0.5 * gauss_val);
}
void runge_kutt1();

void runge_kutt()
{
	std::ofstream file;
	file.open("C:/Users/Артем/Desktop/соду.txt");
	std::ofstream file2;
	file2.open("C:/Users/Артем/Desktop/соду2.txt");
	file << 0 << "  " << 0.5 << std::endl;
	file2 << 0 << "  " << 0.5 << std::endl;
	std::vector<double> gauss_vals_for_an(N);
	std::vector<double> gauss_vals(N);

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(0., 1.);
	double X0 =x0;
	double x1 = x0, x2 = x0, x3 = x0;
	for (int i = 0; i < N; i++)
	{
		double gauss_val = distribution(generator) * sqrt(dt);

		x1 = X0 + dt * (0 * A(x1) + 0 * A(x2) + 0 * A(x3)) + gauss_val * (0 * B(x1) + 0 * B(x2) + 0 * B(x3));
		gauss_val = distribution(generator) * sqrt(dt);
		x2 = X0 + dt * (2./3 * A(x1) + 0 * A(x2) + 0 * A(x3)) + gauss_val * (2./3 * B(x1) + 0 * B(x2) + 0 * B(x3));
		gauss_val = distribution(generator) * sqrt(dt);
		x3 = X0 + dt * (-1 * A(x1) + 1 * A(x2) + 0 * A(x3)) + gauss_val * (-1 * B(x1) + 1 * B(x2) + 0 * B(x3));

		gauss_vals[i] =  distribution(generator)* sqrt(dt);
		double x = X0 + dt * (0 * A(x1) + 3./4 * A(x2) + 1./4 * A(x3)) + gauss_vals[i] * (0 * B(x1) + 3. / 4 * B(x2) + 1. / 4 * B(x3));

		file << left + i * dt << "  " << x << std::endl;
		X0 = x;

//		for (int j = 1; j < i + 1; ++j)
	//		gauss_vals_for_an[i] += gauss_vals[j-1];

		for (int j = 1; j < i+1 ; ++j)
				gauss_vals_for_an[i] += gauss_vals[j];

		file2 << left + dt * i << " " << foo2(left + dt * i, gauss_vals_for_an[i]) << std::endl;
	}
}
void runge_kutt_for_system()
{
	std::ofstream file1;
	file1.open("C:/Users/Артем/Desktop/сур1.txt");
	std::ofstream file2;
	file2.open("C:/Users/Артем/Desktop/сур2.txt");

	std::vector<double> gauss_vals_for_an(N);
	std::vector<double> gauss_vals(N);
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(0., 1.);

	double X01 = x01, X02 = x02;
	double x11 = X01, x12 = X01, x13 = X01, x21 = X02, x22 = X02, x23 = X02;
	for (int i = 0; i < N; i++)
	{
		double gauss_val = distribution(generator) * sqrt(dt);

		x11 = X01 + dt * (0 * A1(x11, x21) + 0 * A1(x12, x22) + 0 * A1(x13, x23)) + gauss_val * (0 * B1(x11, x21) + 0 * B1(x12, x22) + 0 * B1(x13, x23));
		x21 = X02 + dt * (0 * A2(x11, x21) + 0 * A2(x12, x22) + 0 * A2(x13, x23)) + gauss_val * (0 * B2(x11, x21) + 0 * B2(x12, x22) + 0 * B2(x13, x23));

		x12 = X01 + dt * (2. / 3 * A1(x11, x21) + 0 * A1(x12, x22) + 0 * A1(x13, x23)) + gauss_val * (2. / 3 * B1(x11, x21) + 0 * B1(x12, x22) + 0 * B1(x13, x23));
		x22 = X02 + dt * (2. / 3 * A2(x11, x21) + 0 * A2(x12, x22) + 0 * A2(x13, x23)) + gauss_val * (2. / 3 * B2(x11, x21) + 0 * B2(x12, x22) + 0 * B2(x13, x23));

		x13 = X01 + dt * (-1 * A1(x11, x21) + 1 * A1(x12, x22) + 0 * A1(x13, x23)) + gauss_val * (-1 * B1(x11, x21) + 1 * B1(x12, x22) + 0 * B1(x13, x23));
		x23 = X02 + dt * (-1 * A2(x11, x21) + 1 * A2(x12, x22) + 0 * A2(x13, x23)) + gauss_val * (-1 * B2(x11, x21) + 1 * B2(x12, x22) + 0 * B2(x13, x23));

		gauss_vals[i] = distribution(generator) * sqrt(dt);
		double x1 = X01 + dt * (0 * A1(x11, x21) + 3. / 4 * A1(x12, x22) + 1. / 4 * A1(x13, x23)) + gauss_vals[i] * (0 * B1(x11, x21) + 3. / 4 * B1(x12, x22) + 1. / 4 * B1(x13, x23));
		double x2 = X02 + dt * (0 * A2(x11, x21) + 3. / 4 * A2(x12, x22) + 1. / 4 * A2(x13, x23)) + gauss_vals[i] * (0 * B2(x11, x21) + 3. / 4 * B2(x12, x22) + 1. / 4 * B2(x13, x23));

		file1 << left + i * dt << "  " << x1 << std::endl;
		file2 << left + i * dt << "  " << x2 << std::endl;
		//file2 << left + i * dt << "  " << foo2(left + i * dt, x2) << std::endl;
		X01 = x1;
		X02 = x2;

	}
	return;
}
void eiler()
{
	std::ofstream file;
	file.open("C:/Users/Артем/Desktop/соду.txt");
	std::ofstream file2;
	file2.open("C:/Users/Артем/Desktop/соду2.txt");
	//file << 0 << "  " << 0.5 << std::endl;
	//file2 << 0 << "  " << 0.5 << std::endl;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(0., 1.);
	std::vector<double> gauss_vals(N_for_an);
	std::vector<double> gauss_vals_for_an(N_for_an);
	for (size_t i = 0; i < N_for_an; ++i)
	{
		gauss_vals[i] = distribution(generator);
		for (int j = 0; j < i+1; ++j)
			gauss_vals_for_an[i] += gauss_vals[j]*sqrt(Dt);
	}
	double x = 0;
	double x00 = x0;
	double xt = 0;
	double xan = 0;
	for (int i = 0; i < N; ++i)
	{
		int t = i * N_for_an / N;
		double temp=0;
		if(i>0)
		for (int j = (i-1) * N_for_an / N; j < i * N_for_an / N; ++j)
			temp += gauss_vals[j] * sqrt(Dt);
		x = foo(temp, x00);
		file << left + dt * i << "  " << x << std::endl;
		x00 = x;
		if (i == N / 2) sr1 += x;
	}

	for (int i = 0; i < N_for_an; ++i)
	{
		x = foo2(left + Dt * i, gauss_vals_for_an[i]);
		file2 << left + Dt * i << " " << x << std::endl;
		if (i == N / 2) sr2 += x;
	}

	//error += fabs(xan - xt);
}



int main()

{
	//int G = 1000;
	//std::ofstream file;
	//file.open("C:/Users/Артем/Desktop/соду.txt");
	//for (int j = 0; j < 5; ++j)
	//{
	//	for (int i = 0; i < G; ++i)
	//		runge_kutt1();

	//	std::cout << "dt = " << log(dt) << " er = " << log(fabs(sr1/G - sr2/G)) << std::endl;
	//	file << log(dt) << "   " << log(fabs(sr1 / G - sr2 / G)) << std:: endl;
	////   std::cout << "dt = " << log(dt) << " er = " << log(fabs(error/G)) << std::endl;
 //    //  file << log(dt) << "   " << log(fabs(error/G)) << std:: endl;

	//	error = 0;
	//	dt = dt / 2;
	//	sr1 = 0;
	//	sr2 = 0;
	//}




	/*std::ofstream file;
	file.open("C:/Users/Артем/Desktop/соду.txt");
	std::ofstream file2;
	file2.open("C:/Users/Артем/Desktop/соду2.txt");

	std::random_device rd;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0., 1.);
	std::vector<double> gauss_vals(N);
	std::vector<double> wiener_process(N);
	std::vector<double> Dt(N);*/

	runge_kutt_for_system();
	//runge_kutt();
	//runge_kutt1();
	//eiler();

	







}

void runge_kutt1()
{
	//std::ofstream file;
	//file.open("C:/Users/Артем/Desktop/соду.txt");
	//std::ofstream file2;
	//file2.open("C:/Users/Артем/Desktop/соду2.txt");
//	file << 0 << "  " << 0.5 << std::endl;
//	file2 << 0 << "  " << 0.5 << std::endl;
	std::vector<double> gauss_vals_for_an(N_for_an);
	std::vector<double> gauss_vals(N_for_an);

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(0., 1.);
	double X0 = x0;
	double x1 = x0, x2 = x0, x3 = x0;

	for (size_t i = 0; i < N_for_an; ++i)
	{
		gauss_vals[i] = distribution(generator);
		for (int j = 0; j < i + 1; ++j)
			gauss_vals_for_an[i] += gauss_vals[j] * sqrt(Dt);
	}

	for (int i = 0; i < N; i++)
	{
		double gauss_val = distribution(generator) * sqrt(dt);
		x1 = X0 + dt * (0 * A(x1) + 0 * A(x2) + 0 * A(x3)) + gauss_val * (0 * B(x1) + 0 * B(x2) + 0 * B(x3));
		gauss_val = distribution(generator) * sqrt(dt);
		x2 = X0 + dt * (2. / 3 * A(x1) + 0 * A(x2) + 0 * A(x3)) + gauss_val * (2. / 3 * B(x1) + 0 * B(x2) + 0 * B(x3));
		gauss_val = distribution(generator) * sqrt(dt);
		x3 = X0 + dt * (-1 * A(x1) + 1 * A(x2) + 0 * A(x3)) + gauss_val * (-1 * B(x1) + 1 * B(x2) + 0 * B(x3));

		double temp = 0;
		if (i > 0)
		for (int j = (i - 1) * N_for_an / N; j < i * N_for_an / N; ++j)
			temp += gauss_vals[j] * sqrt(Dt);

		//gauss_vals[i] = distribution(generator) * sqrt(dt);
		double x = X0 + dt * (0 * A(x1) + 3. / 4 * A(x2) + 1. / 4 * A(x3)) + temp * (0 * B(x1) + 3. / 4 * B(x2) + 1. / 4 * B(x3));
		if (i == N / 2) {
			sr1 += x;
			xt = x;
		}
	//	file << left + i * dt << "  " << x << std::endl;
		X0 = x;

		//		for (int j = 1; j < i + 1; ++j)
			//		gauss_vals_for_an[i] += gauss_vals[j-1];

	/*	for (int j = 1; j < i + 1; ++j)
			gauss_vals_for_an[i] += gauss_vals[j];*/

		//file2 << left + dt * i << " " << foo2(left + dt * i, gauss_vals_for_an[i]) << std::endl;
	}
	for (int i = 0; i < N_for_an; ++i)
	{
		double x = foo2(left + Dt * i, gauss_vals_for_an[i]);
	//	file2 << left + Dt * i << " " << x << std::endl;
		if (i == N / 2) {
			sr2 += x;
			xan = x;
		}
	}
	error += fabs(xan - xt);
}