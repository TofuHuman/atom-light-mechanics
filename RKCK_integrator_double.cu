#include <stdio.h>
#include <iostream>
#include <limits>
#include <fstream>
#include <string>

#include <curand.h>
#include<cmath>

typedef std::numeric_limits< double > dbl;

// simulation parameters
const double tol = 0.01; 
const double T_stop = 100000;
const size_t N = 1000;

// basic constants
const double kB = 1.38065e-23;
const double g = 9.805e-6;
const double pi = 3.14159;

// Fiber and Field Parameters 
const double r_casimir_cutoff = 0.1;


//******************************************************//
//******************************************************//
// // T. Yoon et al. PRA, 2019
// // T. Yoon et al. J. Phys, 2020
// // const char atom_species = 'Cs';

const double r_hcpcf = 3.75 - r_casimir_cutoff;
const double alpha = 9.4657e-18; 
const double m = 2.2069e-25;

const double P = 0.4e-7;
const double w0 = 2.75;
const double wvlngth = 0.935; 
const double zR = 25.40994058050568;

const double Temp = 32e-6;
const double cx = 0;
const double cy = 0;
const double cz = 5000;
const double R = 1500;

//******************************************************//
//******************************************************//
// // M. Bajcsy et al. PRA, 2011
// // const char atom_species = 'Rb';

// const double r_hcpcf = 3.5 - r_casimir_cutoff;
// const double alpha = 3.038831243e-17;
// const double m = 1.44316060e-25;

// const double P = 0.25e-7;
// const double w0 = 2;
// const double wvlngth = 0.802;
// const double zR = 15.668791289724654;

// const double Temp = 40e-6;
// const double cx = 0;
// const double cy = 0;
// const double cz = 6300;
// const double R = 340;

//******************************************************//
//******************************************************//
// // A. P Hilton et al. PRApplied, 2018
// // const char atom_species = 'Rb';

// const double r_hcpcf = 22.5 - r_casimir_cutoff;
// const double m = 7.90716269690906e-17;

// const double P = 10e-7;
// const double w0 = 16.5;
// const double wvlngth = 0.802;
// const double zR = 1073.147553249462;

// const double Temp = 5e-6;
// const double cx = 0;
// const double cy = 0;
// const double cz = 25000;
// const double R = 500;  // Gaussian MOT

//******************************************************//
//******************************************************//
// // Yang et al. Fibers, 2020
// // const char atom_species = 'Rb';

// const double r_hcpcf = 32 - r_casimir_cutoff;
// const double m = 1.44316060e-25;
// const double alpha = 1.1727029545230675e-17;

// const double P = 5e-7;
// const double w0 = 22;
// const double wvlngth = 0.821;
// const double zR = 1852.0473134439221;

// const double Temp = 10e-6;
// const double cx = 0;
// const double cy = 0;
// const double cz = 5000;
// const double R = 1000; // Gaussian MOT

//******************************************************//
//******************************************************//

// particle size 


__device__ void trial(double z, int &statusi)
{   
    if (z > 0) {
        statusi = 0;
        
    }
    
    else {
        statusi = 1;
    }
    
}


__device__ void a_dipole(double x, double y, double z, double &ax, double &ay, double &az)
{
    double w = 0;
    double gaussian = 0;
    double intensity = 0;
    
    double ax_dipole = 0;
    double ay_dipole = 0;
    double az_dipole = 0;
    
    double r = sqrt(pow(x,2) + pow(y,2));
    double I0 = (2*P)/(pi*pow(w0,2));
    
    if (z > 0) {
        w = w0*sqrt(1 + pow((z/zR),2));
        gaussian = exp(-(2*(pow(r,2)))/(pow(w,2)));
        intensity = I0*pow((w0/w),2)*gaussian;
        ax_dipole = -4*x*alpha*intensity/(m*pow(w,2));
        ay_dipole = -4*y*alpha*intensity/(m*pow(w,2));
        az_dipole = alpha*intensity*(4*(z/(pow(zR,2)))*pow((w0/w),4)*((pow(x,2) + pow(y,2))/pow(w0,2)) - (2*z/(pow(zR,2)))*pow((w0/w),2))/m;
    }
    
    else {
        w = w0;
        gaussian = exp(-(2*(pow(r,2)))/(pow(w,2)));
        intensity = I0*pow((w0/w),2)*gaussian;
        ax_dipole = -4*x*alpha*intensity/(m*pow(w,2));
        ay_dipole = -4*y*alpha*intensity/(m*pow(w,2));
        az_dipole = 0;
    }
    
    ax = ax_dipole;
    ay = ay_dipole;
    az = az_dipole - g;
}

__global__ void RKCK_update(double *px, double *py, double *pz, double *vx, double *vy, double *vz, double *ax, double *ay, double *az, double* t, double *dt, double *err, int *statusi, double* k1px, double* k1py, double* k1pz, double* k1vx, double* k1vy, double* k1vz, double* k2px, double* k2py, double* k2pz, double* k2vx, double* k2vy, double* k2vz, double* k3px, double* k3py, double* k3pz, double* k3vx, double* k3vy, double* k3vz, double* k4px, double* k4py, double* k4pz, double* k4vx, double* k4vy, double* k4vz, double* k5px, double* k5py, double* k5pz, double* k5vx, double* k5vy, double* k5vz, double* k6px, double* k6py, double* k6pz, double* k6vx, double* k6vy, double* k6vz, double* dpx, double* dpy, double* dpz, double* dpx_1, double* dpy_1, double* dpz_1)
{   
    
    double px_temp, py_temp, pz_temp, vx_temp, vy_temp, vz_temp, ax_temp, ay_temp, az_temp; 
    double errx, erry, errz;
    double z_stop = -100;
    
    double B[6][5] = {
        {0, 0, 0, 0, 0}, 
        {1.0/5.0, 0, 0, 0, 0},
        {3.0/40.0, 9.0/40.0, 0, 0, 0},
        {3.0/10.0, -9.0/10.0, 6.0/5.0, 0, 0},
        {-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0, 0},
        {1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0}
    };
    
    double C[] = {37.0/378.0, 0, 250.0/621.0, 125.0/594.0, 0, 512.0/1771.0};
    double C_1[] = {2825.0/27648.0, 0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1./4.0};
        
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int particle = index; particle < N; particle += stride){

        for (int T_timer = 0; (pz[particle] > z_stop && T_timer < T_stop); T_timer++){ //&& T_timer < 50000

            err[particle] = 2*tol;
                        
            for (int T_error = 0; err[particle] > tol; T_error++){
                
                // 1
                k1vx[particle] = ax[particle]*dt[particle];
                k1px[particle] = vx[particle]*dt[particle];
                
                k1vy[particle] = ay[particle]*dt[particle];
                k1py[particle] = vy[particle]*dt[particle];             
                
                k1vz[particle] = az[particle]*dt[particle];
                k1pz[particle] = vz[particle]*dt[particle];
                
                // 2
                px_temp = px[particle] + B[1][0]*k1px[particle];
                py_temp = py[particle] + B[1][0]*k1py[particle];
                pz_temp = pz[particle] + B[1][0]*k1pz[particle];
                vx_temp = vx[particle] + B[1][0]*k1vx[particle];
                vy_temp = vy[particle] + B[1][0]*k1vy[particle];
                vz_temp = vz[particle] + B[1][0]*k1vz[particle];
                a_dipole(px_temp, py_temp, pz_temp, ax_temp, ay_temp, az_temp);
                
                k2vx[particle] = ax_temp*dt[particle]; 
                k2px[particle] = vx_temp*dt[particle];
                
                k2vy[particle] = ay_temp*dt[particle]; 
                k2py[particle] = vy_temp*dt[particle];
                
                k2vz[particle] = az_temp*dt[particle]; 
                k2pz[particle] = vz_temp*dt[particle];
                
                // 3
                px_temp = px[particle] + B[2][0]*k1px[particle] + B[2][1]*k2px[particle];
                py_temp = py[particle] + B[2][0]*k1py[particle] + B[2][1]*k2py[particle];
                pz_temp = pz[particle] + B[2][0]*k1pz[particle] + B[2][1]*k2pz[particle];
                vx_temp = vx[particle] + B[2][0]*k1vx[particle] + B[2][1]*k2vx[particle];
                vy_temp = vy[particle] + B[2][0]*k1vy[particle] + B[2][1]*k2vy[particle];
                vz_temp = vz[particle] + B[2][0]*k1vz[particle] + B[2][1]*k2vz[particle];
                a_dipole(px_temp, py_temp, pz_temp, ax_temp, ay_temp, az_temp);
                
                k3vx[particle] = ax_temp*dt[particle]; 
                k3px[particle] = vx_temp*dt[particle];
                
                k3vy[particle] = ay_temp*dt[particle]; 
                k3py[particle] = vy_temp*dt[particle];
                
                k3vz[particle] = az_temp*dt[particle]; 
                k3pz[particle] = vz_temp*dt[particle];
                
                // 4
                px_temp = px[particle] + B[3][0]*k1px[particle] + B[3][1]*k2px[particle] + B[3][2]*k3px[particle];
                py_temp = py[particle] + B[3][0]*k1py[particle] + B[3][1]*k2py[particle] + B[3][2]*k3py[particle];
                pz_temp = pz[particle] + B[3][0]*k1pz[particle] + B[3][1]*k2pz[particle] + B[3][2]*k3pz[particle];
                vx_temp = vx[particle] + B[3][0]*k1vx[particle] + B[3][1]*k2vx[particle] + B[3][2]*k3vx[particle];
                vy_temp = vy[particle] + B[3][0]*k1vy[particle] + B[3][1]*k2vy[particle] + B[3][2]*k3vy[particle];
                vz_temp = vz[particle] + B[3][0]*k1vz[particle] + B[3][1]*k2vz[particle] + B[3][2]*k3vz[particle];
                a_dipole(px_temp, py_temp, pz_temp, ax_temp, ay_temp, az_temp);
                
                k4vx[particle] = ax_temp*dt[particle]; 
                k4px[particle] = vx_temp*dt[particle];
                
                k4vy[particle] = ay_temp*dt[particle]; 
                k4py[particle] = vy_temp*dt[particle];
                
                k4vz[particle] = az_temp*dt[particle]; 
                k4pz[particle] = vz_temp*dt[particle];
                
                // 5
                px_temp = px[particle] + B[4][0]*k1px[particle] + B[4][1]*k2px[particle] + B[4][2]*k3px[particle] + B[4][3]*k4px[particle];
                py_temp = py[particle] + B[4][0]*k1py[particle] + B[4][1]*k2py[particle] + B[4][2]*k3py[particle] + B[4][3]*k4py[particle];
                pz_temp = pz[particle] + B[4][0]*k1pz[particle] + B[4][1]*k2pz[particle] + B[4][2]*k3pz[particle] + B[4][3]*k4pz[particle];
                vx_temp = vx[particle] + B[4][0]*k1vx[particle] + B[4][1]*k2vx[particle] + B[4][2]*k3vx[particle] + B[4][3]*k4vx[particle];
                vy_temp = vy[particle] + B[4][0]*k1vy[particle] + B[4][1]*k2vy[particle] + B[4][2]*k3vy[particle] + B[4][3]*k4vy[particle];
                vz_temp = vz[particle] + B[4][0]*k1vz[particle] + B[4][1]*k2vz[particle] + B[4][2]*k3vz[particle] + B[4][3]*k4vz[particle];
                a_dipole(px_temp, py_temp, pz_temp, ax_temp, ay_temp, az_temp);
                
                k5vx[particle] = ax_temp*dt[particle]; 
                k5px[particle] = vx_temp*dt[particle];
                
                k5vy[particle] = ay_temp*dt[particle]; 
                k5py[particle] = vy_temp*dt[particle];
                
                k5vz[particle] = az_temp*dt[particle]; 
                k5pz[particle] = vz_temp*dt[particle];
                
                // 6
                px_temp = px[particle] + B[5][0]*k1px[particle] + B[5][1]*k2px[particle] + B[5][2]*k3px[particle] + B[5][3]*k4px[particle] + B[5][4]*k5px[particle];
                py_temp = py[particle] + B[5][0]*k1py[particle] + B[5][1]*k2py[particle] + B[5][2]*k3py[particle] + B[5][3]*k4py[particle] + B[5][4]*k5py[particle];
                pz_temp = pz[particle] + B[5][0]*k1pz[particle] + B[5][1]*k2pz[particle] + B[5][2]*k3pz[particle] + B[5][3]*k4pz[particle] + B[5][4]*k5pz[particle];
                vx_temp = vx[particle] + B[5][0]*k1vx[particle] + B[5][1]*k2vx[particle] + B[5][2]*k3vx[particle] + B[5][3]*k4vx[particle] + B[5][4]*k5vx[particle];
                vy_temp = vy[particle] + B[5][0]*k1vy[particle] + B[5][1]*k2vy[particle] + B[5][2]*k3vy[particle] + B[5][3]*k4vy[particle] + B[5][4]*k5vy[particle];
                vz_temp = vz[particle] + B[5][0]*k1vz[particle] + B[5][1]*k2vz[particle] + B[5][2]*k3vz[particle] + B[5][3]*k4vz[particle] + B[5][4]*k5vz[particle];
                a_dipole(px_temp, py_temp, pz_temp, ax_temp, ay_temp, az_temp);

                k5vx[particle] = ax_temp*dt[particle]; 
                k5px[particle] = vx_temp*dt[particle];
                
                k5vy[particle] = ay_temp*dt[particle]; 
                k5py[particle] = vy_temp*dt[particle];
                
                k5vz[particle] = az_temp*dt[particle]; 
                k5pz[particle] = vz_temp*dt[particle];
                
                // comparison and error check 
                dpx[particle] = C[0]*k1px[particle]+ C[1]*k2px[particle] + C[2]*k3px[particle] + C[3]*k4px[particle] + C[4]*k5px[particle] + C[5]*k6px[particle];
                dpy[particle] = C[0]*k1py[particle]+ C[1]*k2py[particle] + C[2]*k3py[particle] + C[3]*k4py[particle] + C[4]*k5py[particle] + C[5]*k6py[particle];
                dpz[particle] = C[0]*k1pz[particle]+ C[1]*k2pz[particle] + C[2]*k3pz[particle] + C[3]*k4pz[particle] + C[4]*k5pz[particle] + C[5]*k6pz[particle];
                
                dpx_1[particle] = C_1[0]*k1px[particle]+ C_1[1]*k2px[particle] + C_1[2]*k3px[particle] + C_1[3]*k4px[particle] + C_1[4]*k5px[particle] + C_1[5]*k6px[particle];
                dpy_1[particle] = C_1[0]*k1py[particle]+ C_1[1]*k2py[particle] + C_1[2]*k3py[particle] + C_1[3]*k4py[particle] + C_1[4]*k5py[particle] + C_1[5]*k6py[particle];
                dpz_1[particle] = C_1[0]*k1pz[particle]+ C_1[1]*k2pz[particle] + C_1[2]*k3pz[particle] + C_1[3]*k4pz[particle] + C_1[4]*k5pz[particle] + C_1[5]*k6pz[particle];
                
                // ad-hoc trial and error parameters 
                errx =  1e-6 + (abs(dpx[particle] - dpx_1[particle]));
                erry =  1e-6 + (abs(dpy[particle] - dpy_1[particle]));
                errz =  1e-6 + (abs(dpz[particle] - dpz_1[particle]));
                
                err[particle] = errx + erry + errz;
                dt[particle] = 0.9 * dt[particle] * pow(tol/err[particle], 1.0/5.0);
            }
            
            // update position velocity and acceleration when within error
            px[particle] = px[particle] + dpx[particle];
            py[particle] = py[particle] + dpy[particle];
            pz[particle] = pz[particle] + dpz[particle];
            
            vx[particle] = vx[particle] + C[0]*k1vx[particle]+ C[1]*k2vx[particle] + C[2]*k3vx[particle] + C[3]*k4vx[particle] + C[4]*k5vx[particle] + C[5]*k6vx[particle];
            vy[particle] = vy[particle] + C[0]*k1vy[particle]+ C[1]*k2vy[particle] + C[2]*k3vy[particle] + C[3]*k4vy[particle] + C[4]*k5vy[particle] + C[5]*k6vy[particle];
            vz[particle] = vz[particle] + C[0]*k1vz[particle]+ C[1]*k2vz[particle] + C[2]*k3vz[particle] + C[3]*k4vz[particle] + C[4]*k5vz[particle] + C[5]*k6vz[particle];
            
            a_dipole(px[particle], py[particle], pz[particle], ax[particle], ay[particle], az[particle]);
            t[particle] = t[particle] + dt[particle];

        }    
    }
    
}   


void initialize_MB_cloud(double *px, double *py, double *pz, double *vx, double *vy, double *vz, double *ax, double *ay, double *az, double *t, double *dt, int *statusi){
    
    // initializing MB cloud functions
    double *phi, *costheta, *u, *therm; 
    double theta, r;
    
    int N_therm = 10*N;  // why?
    
    phi = (double *) malloc(N*sizeof(double));
    costheta = (double *) malloc(N*sizeof(double));
    u = (double *) malloc(N*sizeof(double));
    therm = (double *) malloc(N_therm*sizeof(double));
    
    
    // random number generation   
    double *d_phi, *d_costheta, *d_u, *d_therm;
    cudaMalloc(&d_phi, N*sizeof(double));
    cudaMalloc(&d_costheta, N*sizeof(double));
    cudaMalloc(&d_u, N*sizeof(double));
    cudaMalloc(&d_therm, N_therm*sizeof(double));
    
    unsigned int seeder, seeder1, seeder2, seeder3;
    seeder = 1234ULL;
    seeder1 = 1234ULL;
    seeder2 = 1234ULL;
    seeder3 = 1234ULL;
    
    // time_t seeder;
    // time(&seeder);
    // srand((unsigned int) seeder);
    
    // time_t seeder1;
    // time(&seeder1);
    // srand((unsigned int) seeder1);
    
    // time_t seeder2;
    // time(&seeder2);
    // srand((unsigned int) seeder2);
    
    // time_t seeder3;
    // time(&seeder3);
    // srand((unsigned int) seeder3);
    
    // generates N random numbers between 0 and 1
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seeder);
    curandGenerateUniformDouble(gen, d_phi, N);
    cudaMemcpy(phi, d_phi, N*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen);
    cudaFree(d_phi);
    
    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen1, seeder1);
    curandGenerateUniformDouble(gen1, d_costheta, N);
    cudaMemcpy(costheta, d_costheta, N*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen1);
    cudaFree(d_costheta);
    
    curandGenerator_t gen2;
    curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen2, seeder2);
    curandGenerateUniformDouble(gen2, d_u, N);
    cudaMemcpy(u, d_u, N*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen2);
    cudaFree(d_u);
    
    // normal distribution to sample thermal velocities 
    curandGenerator_t gen3;
    curandCreateGenerator(&gen3, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen3, seeder3);
    curandGenerateNormalDouble(gen3, d_therm, N_therm, 0, 1);
    cudaMemcpy(therm, d_therm, N_therm*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen3);
    cudaFree(d_therm);
    
    // initializing (p, v, a, t, status) array values for atoms
    for (int i = 0; i < N; i++){
        
        phi[i] = phi[i]*2*pi;
        costheta[i] = costheta[i]*2 - 1;
        
        theta = acos(costheta[i]);
        //printf("theta = %lf\n", theta);
        r = R*cbrt(u[i]);
        
        px[i] = cx + r*sin(theta)*cos(phi[i]);
        py[i] = cy + r*sin(theta)*sin(phi[i]);
        pz[i] = cz + r*cos(theta);
        
        //printf("pz = %lf\n", pz[i]);
        
        vx[i] = sqrt(kB*Temp/m)*therm[i];
        vy[i] = sqrt(kB*Temp/m)*therm[i + N];
        vz[i] = sqrt(kB*Temp/m)*therm[i + 2*N];
        
        dt[i] = 0.1;
        t[i] = 0.0;
        statusi[i] = 0;
        
        ax[i] = -g;
        ay[i] = -g;
        az[i] = -g;
    }
    
}

void csv_1D_array_reader(const char * filename, double *array){
    
    std::ifstream inFile (filename);
    std::string data;
    
    int i = 0;
    if (inFile.is_open()){
        
        while(getline(inFile, data)){
            
            std::cout << data << ',';
            const char * c = data.c_str();
            array[i] = std::atof(c);
            i = i + 1; 
        }
        
        inFile.close();
    }
    
}

void csv_1D_array_writer(const char * filename, double *array){
    
    std::ifstream inFile (filename);
    std::string data;
    
    int i = 0;
    if (inFile.is_open()){
        
        while(getline(inFile, data)){
            
            std::cout << data << ',';
            const char * c = data.c_str();
            array[i] = std::atof(c);
            i = i + 1; 
        }
        
        inFile.close();
    }
    
}


int main(void){
    
    cudaDeviceReset();
    
    // initialize 1D arrays for particles
    double *px, *py, *pz, *vx, *vy, *vz, *ax, *ay, *az, *t, *dt;
    int *statusi;
    
    // 1D arrays for position, velocity, acceleration, timer, adaptive step and status for particles 
    px = (double *) malloc(N*sizeof(double));
    py = (double *) malloc(N*sizeof(double));
    pz = (double *) malloc(N*sizeof(double));
    vx = (double *) malloc(N*sizeof(double));
    vy = (double *) malloc(N*sizeof(double));
    vz = (double *) malloc(N*sizeof(double));
    ax = (double *) malloc(N*sizeof(double));
    ay = (double *) malloc(N*sizeof(double));
    az = (double *) malloc(N*sizeof(double));
    dt = (double *) malloc(N*sizeof(double));
    t = (double *) malloc(N*sizeof(double));
    statusi = (int *) malloc(N*sizeof(int));
    
    
    //****************************************
    //****************************************
    // initializing RKCK parameters 
    
    double *k1px, *k1vx, *k2px, *k2vx, *k3px, *k3vx, *k4px, *k4vx, *k5px, *k5vx, *k6vx, *k6px, *dpx, *dpx_1;
    double *k1py, *k1vy, *k2py, *k2vy, *k3py, *k3vy, *k4py, *k4vy, *k5py, *k5vy, *k6vy, *k6py, *dpy, *dpy_1;
    double *k1pz, *k1vz, *k2pz, *k2vz, *k3pz, *k3vz, *k4pz, *k4vz, *k5pz, *k5vz, *k6vz, *k6pz, *dpz, *dpz_1;
    double *err;
    
    // allocate memory to RKCK parameters
    k1px = (double *) malloc(N*sizeof(double));
    k1vx = (double *) malloc(N*sizeof(double));
    k2px = (double *) malloc(N*sizeof(double));
    k2vx = (double *) malloc(N*sizeof(double));
    k3px = (double *) malloc(N*sizeof(double));
    k3vx = (double *) malloc(N*sizeof(double));
    k4px = (double *) malloc(N*sizeof(double));
    k4vx = (double *) malloc(N*sizeof(double));
    k5px = (double *) malloc(N*sizeof(double));
    k5vx = (double *) malloc(N*sizeof(double));
    k6px = (double *) malloc(N*sizeof(double));
    k6vx = (double *) malloc(N*sizeof(double));
    
    k1py = (double *) malloc(N*sizeof(double));
    k1vy = (double *) malloc(N*sizeof(double));
    k2py = (double *) malloc(N*sizeof(double));
    k2vy = (double *) malloc(N*sizeof(double));
    k3py = (double *) malloc(N*sizeof(double));
    k3vy = (double *) malloc(N*sizeof(double));
    k4py = (double *) malloc(N*sizeof(double));
    k4vy = (double *) malloc(N*sizeof(double));
    k5py = (double *) malloc(N*sizeof(double));
    k5vy = (double *) malloc(N*sizeof(double));
    k6py = (double *) malloc(N*sizeof(double));
    k6vy = (double *) malloc(N*sizeof(double));
    
    k1pz = (double *) malloc(N*sizeof(double));
    k1vz = (double *) malloc(N*sizeof(double));
    k2pz = (double *) malloc(N*sizeof(double));
    k2vz = (double *) malloc(N*sizeof(double));
    k3pz = (double *) malloc(N*sizeof(double));
    k3vz = (double *) malloc(N*sizeof(double));
    k4pz = (double *) malloc(N*sizeof(double));
    k4vz = (double *) malloc(N*sizeof(double));
    k5pz = (double *) malloc(N*sizeof(double));
    k5vz = (double *) malloc(N*sizeof(double));
    k6pz = (double *) malloc(N*sizeof(double));
    k6vz = (double *) malloc(N*sizeof(double));
    
    dpx = (double *) malloc(N*sizeof(double));
    dpy = (double *) malloc(N*sizeof(double));
    dpz = (double *) malloc(N*sizeof(double));
    
    dpx_1 = (double *) malloc(N*sizeof(double));
    dpy_1 = (double *) malloc(N*sizeof(double));
    dpz_1 = (double *) malloc(N*sizeof(double));
    
    double tol = 0.00001;
    
    err = (double *) malloc(N*sizeof(double));
    
    for (int i = 0; i < N; i++){

        err[i] = 2*tol;
        k1px[i] = 0;
        k1vx[i] = 0;
        k2px[i] = 0;
        k2vx[i] = 0;
        k3px[i] = 0;
        k3px[i] = 0;
        k4px[i] = 0;
        k4vx[i] = 0;
        k5px[i] = 0;
        k5px[i] = 0;
        k6px[i] = 0;
        k6px[i] = 0;
        
        k1py[i] = 0;
        k1vy[i] = 0;
        k2py[i] = 0;
        k2vy[i] = 0;
        k3py[i] = 0;
        k3py[i] = 0;
        k4py[i] = 0;
        k4vy[i] = 0;
        k5py[i] = 0;
        k5py[i] = 0;
        k6py[i] = 0;
        k6py[i] = 0;
        
        k1pz[i] = 0;
        k1vz[i] = 0;
        k2pz[i] = 0;
        k2vz[i] = 0;
        k3pz[i] = 0;
        k3pz[i] = 0;
        k4pz[i] = 0;
        k4vz[i] = 0;
        k5pz[i] = 0;
        k5pz[i] = 0;
        k6pz[i] = 0;
        k6pz[i] = 0;
        
        dpx[i] = 0;
        dpx_1[i] = 0;
        dpy[i] = 0;
        dpy_1[i] = 0;
        dpz[i] = 0;
        dpz_1[i] = 0;
        
    }
    //****************************************
    //****************************************
    
    //****************************************************************************//
    // sample positions and velocities from a uniform sphere and an MB distribution 
    initialize_MB_cloud(px, py, pz, vx, vy, vz, ax, ay, az, t, dt, statusi);
    //****************************************************************************//
    
    // // stop cout
    // std::streambuf* orig_buf = std::cout.rdbuf();
    // std::cout.rdbuf(NULL);
    
    // //************************************************//
    // // read from python generated data for comparison://
    // csv_1D_array_reader("../other/mb_sphere_data/px.csv", px);
    // csv_1D_array_reader("../other/mb_sphere_data/py.csv", py);
    // csv_1D_array_reader("../other/mb_sphere_data/pz.csv", pz);
    // csv_1D_array_reader("../other/mb_sphere_data/vx.csv", vx);
    // csv_1D_array_reader("../other/mb_sphere_data/vy.csv", vy);
    // csv_1D_array_reader("../other/mb_sphere_data/vz.csv", vz);
    
    // for (int i = 0; i < N; i++){
        
    //     dt[i] = 0.1;

    //     t[i] = 0.0;
    //     statusi[i] = 0;
    
    //     ax[i] = 0;
    //     ay[i] = 0;
    //     az[i] = 0;
    // }
    // //************************************************//
    
    // // resume cout
    // std::cout.rdbuf(orig_buf);
    
    //*****************************************************//
    // manual definition 
    
    // for (int i = 0; i < N; i++){
        
    //     px[i] = 1000 + (i + 1)*pi;
    //     py[i] = 1000 + (i + 1)*pi;
    //     pz[i] = 1000 + (i + 1)*pi;
        
    //     vx[i] = 0;
    //     vy[i] = 0;
    //     vz[i] = 0;
        
    //     ax[i] = 0;
    //     ay[i] = 0;
    //     az[i] = 0;
        
    //     dt[i] = 0.1; 
    //     t[i] = 0;
    //     statusi[i] = 0;
        
    // }
    
    
    
    //************************************************//
    printf("----------------------------\n");
    printf("Before Evolution \n");
    printf("px = %lf\n", px[0]);
    printf("py = %lf\n", py[0]);
    printf("pz = %lf\n", pz[0]);
    
    printf("vx = %lf\n", vx[0]);
    printf("vy = %lf\n", vy[0]);
    printf("vz = %lf\n", vz[0]);
    
    printf("az = %lf\n", az[0]);
    printf("status = %d\n", statusi[0]);
    printf("----------------------------\n");
    std::cout.precision(dbl::digits10);
    std::cout << "py (precision) " << py[0] << std::endl;
    printf("----------------------------\n");
    //************************************************//
    
    
    // //************************************************//
    // printf("----------------------------\n");
    // printf("Before Evolution \n");
    // printf("px = %lf\n", px[1]);
    // printf("py = %lf\n", py[1]);
    // printf("pz = %lf\n", pz[1]);
    
    // printf("vx = %lf\n", vx[1]);
    // printf("vy = %lf\n", vy[1]);
    // printf("vz = %lf\n", vz[1]);
    
    // printf("az = %lf\n", az[0]);
    // printf("status = %d\n", statusi[0]);
    // printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "py (precision) " << py[0] << std::endl;
    // printf("----------------------------\n");
    // //************************************************//
    
    
    
    
    //************************************************//
    printf("Memory Allocation Begin\n");
    
    // device: 1D arrays for position, velocity and acceleration
    double *dev_px, *dev_py, *dev_pz, *dev_vx, *dev_vy, *dev_vz, *dev_ax, *dev_ay, *dev_az, *dev_timer, *dev_dt;
    int *dev_statusi;
    
    // device: allocation of  memory 
    cudaMalloc(&dev_px, N*sizeof(double)); 
    cudaMalloc(&dev_py, N*sizeof(double)); 
    cudaMalloc(&dev_pz, N*sizeof(double)); 
    cudaMalloc(&dev_vx, N*sizeof(double)); 
    cudaMalloc(&dev_vy, N*sizeof(double)); 
    cudaMalloc(&dev_vz, N*sizeof(double)); 
    cudaMalloc(&dev_ax, N*sizeof(double)); 
    cudaMalloc(&dev_ay, N*sizeof(double)); 
    cudaMalloc(&dev_az, N*sizeof(double));
    cudaMalloc(&dev_timer, N*sizeof(double)); 
    cudaMalloc(&dev_dt, N*sizeof(double)); 
    cudaMalloc(&dev_statusi, N*sizeof(int));
    
    
    // device: copying initial values 
    cudaMemcpy(dev_px, px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_py, py, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pz, pz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vx, vx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vy, vy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vz, vz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ax, ax, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ay, ay, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_az, az, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_timer, t, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dt, dt, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_statusi, statusi, N*sizeof(int), cudaMemcpyHostToDevice);
    
    //******************************
    //******************************
    // allocate memory to RKCK parameters
    double *dev_k1px, *dev_k1vx, *dev_k2px, *dev_k2vx, *dev_k3px, *dev_k3vx, *dev_k4px, *dev_k4vx, *dev_k5px, *dev_k5vx, *dev_k6vx, *dev_k6px, *dev_dpx, *dev_dpx_1;
    double *dev_k1py, *dev_k1vy, *dev_k2py, *dev_k2vy, *dev_k3py, *dev_k3vy, *dev_k4py, *dev_k4vy, *dev_k5py, *dev_k5vy, *dev_k6vy, *dev_k6py, *dev_dpy, *dev_dpy_1;
    double *dev_k1pz, *dev_k1vz, *dev_k2pz, *dev_k2vz, *dev_k3pz, *dev_k3vz, *dev_k4pz, *dev_k4vz, *dev_k5pz, *dev_k5vz, *dev_k6vz, *dev_k6pz, *dev_dpz, *dev_dpz_1;
    double *dev_err; 
    
    cudaMalloc(&dev_k1px, N*sizeof(double)); 
    cudaMalloc(&dev_k1vx, N*sizeof(double)); 
    cudaMalloc(&dev_k2px, N*sizeof(double)); 
    cudaMalloc(&dev_k2vx, N*sizeof(double)); 
    cudaMalloc(&dev_k3px, N*sizeof(double)); 
    cudaMalloc(&dev_k3vx, N*sizeof(double)); 
    cudaMalloc(&dev_k4px, N*sizeof(double)); 
    cudaMalloc(&dev_k4vx, N*sizeof(double)); 
    cudaMalloc(&dev_k5px, N*sizeof(double)); 
    cudaMalloc(&dev_k5vx, N*sizeof(double)); 
    cudaMalloc(&dev_k6px, N*sizeof(double)); 
    cudaMalloc(&dev_k6vx, N*sizeof(double)); 
    
    cudaMalloc(&dev_k1py, N*sizeof(double)); 
    cudaMalloc(&dev_k1vy, N*sizeof(double)); 
    cudaMalloc(&dev_k2py, N*sizeof(double)); 
    cudaMalloc(&dev_k2vy, N*sizeof(double)); 
    cudaMalloc(&dev_k3py, N*sizeof(double)); 
    cudaMalloc(&dev_k3vy, N*sizeof(double)); 
    cudaMalloc(&dev_k4py, N*sizeof(double)); 
    cudaMalloc(&dev_k4vy, N*sizeof(double)); 
    cudaMalloc(&dev_k5py, N*sizeof(double)); 
    cudaMalloc(&dev_k5vy, N*sizeof(double)); 
    cudaMalloc(&dev_k6py, N*sizeof(double)); 
    cudaMalloc(&dev_k6vy, N*sizeof(double)); 
    
    cudaMalloc(&dev_k1pz, N*sizeof(double)); 
    cudaMalloc(&dev_k1vz, N*sizeof(double)); 
    cudaMalloc(&dev_k2pz, N*sizeof(double)); 
    cudaMalloc(&dev_k2vz, N*sizeof(double)); 
    cudaMalloc(&dev_k3pz, N*sizeof(double)); 
    cudaMalloc(&dev_k3vz, N*sizeof(double)); 
    cudaMalloc(&dev_k4pz, N*sizeof(double)); 
    cudaMalloc(&dev_k4vz, N*sizeof(double)); 
    cudaMalloc(&dev_k5pz, N*sizeof(double)); 
    cudaMalloc(&dev_k5vz, N*sizeof(double)); 
    cudaMalloc(&dev_k6pz, N*sizeof(double)); 
    cudaMalloc(&dev_k6vz, N*sizeof(double)); 
    
    cudaMalloc(&dev_err, N*sizeof(double)); 
    
    cudaMalloc(&dev_dpx, N*sizeof(double)); 
    cudaMalloc(&dev_dpy, N*sizeof(double)); 
    cudaMalloc(&dev_dpz, N*sizeof(double)); 
    
    cudaMalloc(&dev_dpx_1, N*sizeof(double)); 
    cudaMalloc(&dev_dpy_1, N*sizeof(double)); 
    cudaMalloc(&dev_dpz_1, N*sizeof(double)); 
    
    // memcpy functions
    
    cudaMemcpy(dev_k1px, k1px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k2px, k2px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k3px, k3px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k4px, k4px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k5px, k5px, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k6px, k6px, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_k1py, k1py, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k2py, k2py, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k3py, k3py, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k4py, k4py, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k5py, k5py, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k6py, k6py, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_k1pz, k1pz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k2pz, k2pz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k3pz, k3pz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k4pz, k4pz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k5pz, k5pz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k6pz, k6pz, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_k1vx, k1vx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k2vx, k2vx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k3vx, k3vx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k4vx, k4vx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k5vx, k5vx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k6vx, k6vx, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_k1vy, k1vy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k2vy, k2vy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k3vy, k3vy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k4vy, k4vy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k5vy, k5vy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k6vy, k6vy, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_k1vz, k1vz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k2vz, k2vz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k3vz, k3vz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k4vz, k4vz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k5vz, k5vz, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k6vz, k6vz, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_err, err, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_dpx, dpx, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dpx, dpy, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dpy, dpz, N*sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_dpx_1, dpx_1, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dpx_1, dpy_1, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dpy_1, dpz_1, N*sizeof(double), cudaMemcpyHostToDevice);
    //******************************
    //******************************
    
    
    
    printf("Memory Allocation End \n");
    
    
    // kernel initialization
    // code reference:  https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
    
    int blockSize;      // The launch configurator returned block size 
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    int gridSize;       // The actual grid size needed, based on input size 
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, RKCK_update, 0, N); 
    
    // Round up according to array size 
    gridSize = (N + blockSize - 1) / blockSize; 
    
    printf("Grid Size (GPU, 1D Arrays) = %d\n", gridSize); 
    printf("Block Size (GPU, 1D Arrays) = %d\n", blockSize); 
    
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    
    
    // testing timing of one verlet update of N particles
    cudaEventRecord(cuda_start);
    RKCK_update<<<gridSize, blockSize>>>(dev_px, dev_py, dev_pz, dev_vx, dev_vy, dev_vz, dev_ax, dev_ay, dev_az, dev_timer, dev_dt, dev_err, dev_statusi, dev_k1px, dev_k1py, dev_k1pz, dev_k1vx, dev_k1vy, dev_k1vz, dev_k2px, dev_k2py, dev_k2pz, dev_k2vx, dev_k2vy, dev_k2vz, dev_k3px, dev_k3py, dev_k3pz, dev_k3vx, dev_k3vy, dev_k3vz, dev_k4px, dev_k4py, dev_k4pz, dev_k4vx, dev_k4vy, dev_k4vz, dev_k5px, dev_k5py, dev_k5pz, dev_k5vx, dev_k5vy, dev_k5vz, dev_k6px, dev_k6py, dev_k6pz, dev_k6vx, dev_k6vy, dev_k6vz, dev_dpx, dev_dpy, dev_dpz, dev_dpx_1,dev_dpy_1, dev_dpz_1);
    cudaEventRecord(cuda_stop);
    
    cudaEventSynchronize(cuda_stop);
    float cuda_diffs = 0;
    cudaEventElapsedTime(&cuda_diffs, cuda_start, cuda_stop);
    cuda_diffs = cuda_diffs/1000.0;
    printf("GPU Time for one Verlet update (GPU, 1D Arrays) = %lf\n", cuda_diffs);
    
    // copying data back to CPU
    cudaMemcpy(statusi, dev_statusi, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(t, dev_timer, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(px, dev_px, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(py, dev_py, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pz, dev_pz, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx, dev_vx, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy, dev_vy, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz, dev_vz, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(az, dev_az, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(err, dev_err, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dt, dev_dt, N*sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(dpy, dev_dpy, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dpz, dev_dpz, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dpx, dev_dpx, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(k1px, dev_k1px, N*sizeof(double), cudaMemcpyDeviceToHost);
    
    //************************************************//
    printf("----------------------------\n");
    printf("After Evolution:\n");
    printf("px = %lf\n", px[0]);
    printf("py = %lf\n", py[0]);
    printf("pz = %lf\n", pz[0]);
    printf("vx = %lf\n", vx[0]);
    printf("vy = %lf\n", vy[0]);
    printf("vz = %lf\n", vz[0]);
    printf("az = %lf\n", az[0]);
    printf("status = %d\n", statusi[0]);
    // printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "k1px (precision) " << k1px[0] << std::endl;
    // printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "dpx (precision) " << dpx[0] << std::endl;
    // printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "dpy (precision) " << dpy[0] << std::endl;
    // printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "dpz (precision) " << dpz[0] << std::endl;
    printf("----------------------------\n");
    std::cout.precision(dbl::digits10);
    std::cout << "dt (precision) " << dt[0] << std::endl;
    printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "err (precision) " << err[0] << std::endl;
    // printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "az (precision) " << az[0] << std::endl;
    // printf("----------------------------\n");
    // std::cout.precision(dbl::digits10);
    // std::cout << "py (precision) " << py[0] << std::endl;
    // printf("----------------------------\n");
    //************************************************//
    
    int total = 0; 
    int counter = 0;
    double loading_efficiency; 
    for (int i = 0; i < N; i++){
        if (pz[i] < 0) {
            total ++; 
            if (pow(px[i], 2) + pow(py[i],2) < pow(r_hcpcf, 2)){
                counter ++;
            }
        }            
    }
    loading_efficiency = (double)counter/N;
    
    printf("Total Number of Atoms = %d\n", total);
    printf("Number of Trapped Atoms = %d\n", counter);
    printf("Loading Efficiency (1e-4) = %lf\n", loading_efficiency*1e4);
    
    
    // clearing memory    
    cudaFree(dev_px);
    cudaFree(dev_py);
    cudaFree(dev_pz);
    cudaFree(dev_vx);
    cudaFree(dev_vy);
    cudaFree(dev_vz);
    cudaFree(dev_ax);
    cudaFree(dev_ay);
    cudaFree(dev_az);
    cudaFree(dev_timer);
    cudaFree(dev_statusi);
    free(px);
    free(py);
    free(pz);
    free(vx);
    free(vy);
    free(vz);
    free(ax);
    free(ay);
    free(az);
    free(t);
    free(statusi);
    
    ///
    free(k1px);
    free(k1vx);
    free(k2px);
    free(k2vx);
    free(k3px);
    free(k3vx);
    free(k4px);
    free(k4vx);
    free(k5px);
    free(k5vx);
    free(k6px);
    free(k6vx);
    
    free(k1py);
    free(k1vy);
    free(k2py);
    free(k2vy);
    free(k3py);
    free(k3vy);
    free(k4py);
    free(k4vy);
    free(k5py);
    free(k5vy);
    free(k6py);
    free(k6vy);
    
    free(k1pz);
    free(k1vz);
    free(k2pz);
    free(k2vz);
    free(k3pz);
    free(k3vz);
    free(k4pz);
    free(k4vz);
    free(k5pz);
    free(k5vz);
    free(k6pz);
    free(k6vz);
    
    free(dpx);
    free(dpx_1);
    free(dpy);
    free(dpy_1);
    free(dpz);
    free(dpz_1);
    
    free(err);
    
    
    cudaFree(dev_k1px);
    cudaFree(dev_k1vx);
    cudaFree(dev_k2px);
    cudaFree(dev_k2vx);
    cudaFree(dev_k3px);
    cudaFree(dev_k3vx);
    cudaFree(dev_k4px);
    cudaFree(dev_k4vx);
    cudaFree(dev_k5px);
    cudaFree(dev_k5vx);
    cudaFree(dev_k6px);
    cudaFree(dev_k6vx);
    
    cudaFree(dev_k1py);
    cudaFree(dev_k1vy);
    cudaFree(dev_k2py);
    cudaFree(dev_k2vy);
    cudaFree(dev_k3py);
    cudaFree(dev_k3vy);
    cudaFree(dev_k4py);
    cudaFree(dev_k4vy);
    cudaFree(dev_k5py);
    cudaFree(dev_k5vy);
    cudaFree(dev_k6py);
    cudaFree(dev_k6vy);
    
    cudaFree(dev_k1pz);
    cudaFree(dev_k1vz);
    cudaFree(dev_k2pz);
    cudaFree(dev_k2vz);
    cudaFree(dev_k3pz);
    cudaFree(dev_k3vz);
    cudaFree(dev_k4pz);
    cudaFree(dev_k4vz);
    cudaFree(dev_k5pz);
    cudaFree(dev_k5vz);
    cudaFree(dev_k6pz);
    cudaFree(dev_k6vz);
    
    cudaFree(dev_dpx);
    cudaFree(dev_dpx_1);
    cudaFree(dev_dpy);
    cudaFree(dev_dpy_1);
    cudaFree(dev_dpz);
    cudaFree(dev_dpz_1);
    
    cudaFree(dev_err);
}