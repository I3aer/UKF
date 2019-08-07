#include <iostream>
#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

typedef struct estimates
{
	VectorXd est;
	MatrixXd cov;
}estimates;

int main()
{
	MatrixXd W_m(5,5);

	for (int i=0;i<5;i++)
		W_m.col(i) << i,0,0,0,0;

	VectorXd L(5);
	L << 0,1,2,3,4;

	cout << W_m << endl;

	W_m.col(2) += W_m.col(1) + L;

	cout << W_m.col(2) << endl;


	// Sample mean and statistics of transformed sigma points
	estimates sys;

	sys.est = VectorXd(5);
	sys.est.setZero();

	sys.cov = MatrixXd(5,5);

	sys.cov = W_m;

	cout << sys.cov << endl;

	sys.cov.setZero();

	cout << "setZero is working?\n" << sys.cov << endl;

	vector<int> w {1,2,3,4,5};

	for (unsigned int i = 0; i<5; ++i)
			sys.est  += w[i]*W_m.col(i);

	cout << "est vector:\n" << sys.est << endl;


	// Innovation between sigma points and sample mean
	VectorXd inno(5);

    w = {1,-1,1,-1,1};
	for(unsigned int i= 0; i<5; ++i)
	{
		inno = (W_m.col(i) - sys.est);
		sys.cov += w[i]*(inno*inno.transpose());
	}

	cout << "sys.cov:\n" << sys.cov << endl;

	return 0;
}
