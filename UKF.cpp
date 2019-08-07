/*
 * UKF.cpp
 *
 *  Created on: Feb 28, 2017
 *      Author: baer
 */
#include "UKF.h"

void UKF::draw_Sigma_Points()
{
	sys_sigma.col(0) = state_est;

	// Compute the matrix-squre root using Cholesky decomposition
	LLT<MatrixXd> lltOfA(state_cov);

	// Obtain the lower triangular matrix from state_cov
	MatrixXd L = lltOfA.matrixL();

	sys_sigma.col(0) = state_est;

	unsigned int idx_c = 0;

	for(unsigned int i=1; i<Nsigma; ++i)
	{
		idx_c = (i-1) % sys_dim;

		if (i > sys_dim)
			sys_sigma.col(i) = sys_sigma.col(0) - L.col(idx_c);
		else
			sys_sigma.col(i) = sys_sigma.col(0) + L.col(idx_c);
	}
}

void UKF::time_Update()
{
	draw_Sigma_Points();

	sys_Transformation();

	// Sample mean and covariance of transformed sigma points
	EST sys_sample;

    sys_sample.est = VectorXd(sys_dim);

    sys_sample.cov = MatrixXd(sys_dim,sys_dim);

	sys_sample = compute_Statistics(sys_sigma,sys_dim,sys_noise_cov);

	state_est = sys_sample.est;
	state_cov = sys_sample.cov;
}

void UKF::meas_Update(VectorXd y)
{
	unsigned int meas_dim = y.rows();

	// Sample mean and covariance of transformed sigma points
	EST meas_sample;

	meas_sample.est = VectorXd(meas_dim);
	meas_sample.cov = MatrixXd(meas_dim ,meas_dim);

	meas_Transformation();

	meas_sample = compute_Statistics(meas_sigma,meas_dim,meas_noise_cov);

	// Compute the cross covariance Pxy
    MatrixXd Pxy(sys_dim,meas_dim);

    // Innovations between sigma points and sample mean
    VectorXd sys_inno(sys_dim);

    VectorXd meas_inno(meas_dim);

    Pxy.setZero();
    for(unsigned int i= 0; i<Nsigma; ++i)
    {
    	sys_inno = sys_sigma.col(i) - state_est;

    	meas_inno = meas_sigma.col(i) - meas_sample.est;

    	Pxy += W_c[i]*(sys_inno*meas_inno.transpose());
    }

	//Execute Kalman update step:
	MatrixXd G = Pxy*meas_sample.cov.inverse();

	state_est = state_est + G*(meas_sample.est -y);

	state_cov = state_cov - G*meas_sample.cov*G.transpose();
}

EST UKF::compute_Statistics(MatrixXd sigma, unsigned int dim, MatrixXd noise_cov)
{
 	EST sample;
	sample.est = VectorXd (dim);    // sample mean
	sample.cov = MatrixXd(dim,dim); // sample covariance

	sample.est .setZero();
	for (unsigned int i = 0; i<Nsigma; ++i)
		sample.est  += W_m[i]*sigma.col(i);

	// Innovation between sigma points and sample mean
	VectorXd inno(dim);

	sample.cov.setZero();
	for(unsigned int i= 0; i<Nsigma; ++i)
	{
		inno = sigma.col(i) - sample.est;
		sample.cov += W_c[i]*(inno*inno.transpose());
	}

	// Contribution of additive noise
	sample.cov +=  noise_cov;

	return sample;

}

void UKF::set_Weights(int L)
{
	//Initialize sigma points at time k=0 and their weights
	for(unsigned int i=0; i<Nsigma; ++i)
	{
		if (i == 0)
		{
			W_m.push_back( lambda/(lambda + sys_dim) );

			W_c.push_back( W_m[i]*(1 - pow(alpha,2) + beta) );
		}
		else
		{
			W_m.push_back( 1/(2*(lambda + sys_dim)) );

			W_c.push_back(W_m[i]);
		}
	}

}

void UKF::sys_Transformation()
{
	// Propagate sigma points through nonlinear system model
	for(unsigned int i=0; i<Nsigma;++i)
		sys_sigma.col(i) = (*sys_func)(sys_sigma.col(i));
}

void UKF::meas_Transformation()
{
	// Propagate sigma points through nonlinear measurement model
	for(unsigned int i=0; i<Nsigma; ++i)
		meas_sigma.col(i) = (*meas_func)(sys_sigma.col(i));
}

EST UKF::get_Estimates()
{
	EST sys;

	sys.est = state_est;

	sys.cov = state_cov;

	return sys;
}
