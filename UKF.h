/*
 * UKF.h
 *
 *  Created on: Feb 28, 2017
 *      Author: baer
 */

#ifndef HEADER_UKF_H_
#define HEADER_UKF_H_

#include <Eigen\Dense>
#include <Eigen\Cholesky>

#include <cmath>
#include <vector>

using Eigen::MatrixXd;

using Eigen::VectorXd;

using Eigen::LLT;

using std::vector;

// Structured data type
typedef struct estimates
{
	VectorXd est;
	MatrixXd cov;
}EST;

typedef MatrixXd (*sys_model)(MatrixXd);

typedef MatrixXd (*meas_model)(MatrixXd);


class UKF{

public:
	UKF(VectorXd x0, MatrixXd P0, MatrixXd Q, MatrixXd R, double t_step, sys_model f, meas_model h):
		lambda(pow(alpha,2)*(x0.rows() + kappa) - x0.rows()), deltaT(t_step), sys_dim(x0.rows()),
		Nsigma(2*x0.rows() + 1), state_est(x0), state_cov(P0), sys_noise_cov(R), meas_noise_cov(Q),
		sys_func(f), meas_func(h)
	{
		// Set weights of sigma points
		set_Weights(sys_dim);
	}

	~UKF() {};

	void time_Update();

	void meas_Update(VectorXd);

	EST get_Estimates();

private:
	// Parameters of the UKF
	const double alpha = 1e-4; //spread of sigma points around mean point, 1e-4 <= alpha <=1
	const double kappa = 0;    //secondary scaling parameter
	const double beta = 2.0;   //prior knowledge about dist. of states.
	const double lambda;       //primary scaling parameter

	// Time step between measurements
	const double deltaT;

	// Dimensions of state and measurement vectors
	const unsigned int sys_dim;

	// Number of sigma points for state variable
	const unsigned int Nsigma;

	// State estimate and its covariance matrix
	VectorXd state_est;
	MatrixXd state_cov;

	// Covariance Matrices of (additive) system and measurement noises
	MatrixXd sys_noise_cov;
	MatrixXd meas_noise_cov;

	// Function pointers to system and measurement models
	const sys_model sys_func;
	const meas_model meas_func;

	// Weights of sigma points
	vector<double> W_m; // for sample mean
	vector<double> W_c; // for sample covariance

	// System and measurement sigma points
	MatrixXd sys_sigma;
	MatrixXd meas_sigma;

	// Utility functions
	void set_Weights(int);

	void sys_Transformation();

	void meas_Transformation();

	void draw_Sigma_Points();

	EST compute_Statistics(MatrixXd, unsigned int, MatrixXd);
};

#endif /* HEADER_UKF_H_ */
