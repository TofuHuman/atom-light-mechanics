#include <stdio.h>
#include <iostream>
#include <limits>

#include <curand.h>
#include<cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

typedef std::numeric_limits< double > dbl;

// simulation parameters
const double dt = 0.1;
const int N = 100000;
const int T_max = 100000*(1/dt);
const double energy_error_check = 0.05; 

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

// Big Fiber 
// const double r_hcpcf = 15;
// const double alpha = 9.4657e-18;
// const double m = 2.2069e-25;

// const double P = 1e-7;
// const double w0 = 11.8;
// const double wvlngth = 0.935; 
// const double zR = 467.8453059741634;

// const double Temp = 30e-6;
// const double cx = 0;
// const double cy = 0;
// const double cz = 5000;
// const double R = 1500;

// const double slope = 2000./5000.; 
// const double r_check = 400; 


// // Small Fiber
// const double r_hcpcf = 3.75 - r_casimir_cutoff;
// const double alpha = 9.4657e-18; 
// const double m = 2.2069e-25;

// // scattering rate constants 
// const double beta = 25449.809972971874; 
// const double spont_v_factor = 0.003360217090858776; 
// const double abs_v_factor = 0.0032150016610518837; 

// // field realted constants 
// const double P = 0.5e-7;
// const double w0 = 2.75;
// const double wvlngth = 0.935; 
// const double zR = 25.40994058050568;

// const double Temp = 32e-6;
// const double cx = 0;
// const double cy = 0;
// const double cz = 5000;
// const double R = 1500;

// const double slope = 1000./5000.; 
// const double r_check = 250;


//******************************************************//
//******************************************************//

// // M. Bajcsy et al. PRA, 2011
// // const char atom_species = 'Rb';

// const double r_hcpcf = 3.5 - r_casimir_cutoff;
// const double m = 1.44316060e-25;
// const double alpha = 3.038831243e-17;

// scattering rate constants 
// const double beta = 382499.3875547423; 
// const double spont_v_factor = 0.00578236135589983; 
// const double abs_v_factor = 0.005731739384291537; 

// const double P = 0.25e-7;
// const double w0 = 2;
// const double wvlngth = 0.802;
// const double zR = 15.668791289724654;

// const double Temp = 40e-6;
// const double cx = 0;
// const double cy = 0;
// const double cz = 6300;
// const double R = 340;

// const double slope = 1000./5000.; 
// const double r_check = 250; 

//******************************************************//
//******************************************************//
// // A. P Hilton et al. PRApplied, 2018
// // const char atom_species = 'Rb - 85';

const double r_hcpcf = 22.5 - r_casimir_cutoff;
const double m = 1.409993199e-25;
const double alpha = 7.190132913713667e-17;

// scattering rate constants 
const double beta = 3015911.2228300544; 
const double spont_v_factor = 0.005918379158044901; 
const double abs_v_factor = 0.005901520518109168; 

const double P = 10e-7;
const double w0 = 16.5;
const double wvlngth = 0.79725;
const double zR = 1072.8110378674457;

const double Temp = 5e-6;
const double cx = 0;
const double cy = 0;
const double cz = 25000;
const double R = 1000;  // Gaussian MOT

const double slope = 2000./25000.; 
const double r_check = 500; 

//******************************************************//
//******************************************************//
// // Yang et al. Fibers, 2020
// // const char atom_species = 'Rb - 85';

// const double r_hcpcf = 32 - r_casimir_cutoff;
// const double m = 1.409993199e-25;
// const double alpha = 1.1732431139058278e-17;

// scattering rate constants 
// const double beta = 48021.0283738874; 
// const double spont_v_factor = 0.005918379158044901; 
// const double abs_v_factor = 0.005730800527481771; 

// const double P = 5e-7;
// const double w0 = 22;
// const double wvlngth = 0.821;
// const double zR = 1852.0473134439221;

// const double Temp = 10e-6;
// const double cx = 0;
// const double cy = 0;
// const double cz = 5000;
// const double R = 2000; // Gaussian MOT

// const double slope = 2000./5000.; 
// const double r_check = 500; 

//******************************************************//
//******************************************************//

struct Particles
{
    double p[3];
    double v[3];
    double a[3];
    double t; 
    double energy_initial; 
    double energy_current; 
    double energy_kick_tab;
    int statusi;                                   //  0 -- running 
                                                   //  1 -- z < 0 -- loaded 
                                                   //  2 -- z < 0 -- not loaded
                                                   //  3 -- out of bounds 
    double N_scattering;


};


// force field function 
__device__ void a_dipole(double x, double y, double z, double &ax, double &ay, double &az, int &statusi, double& OPE)
{   
    double w = 0;
    double gaussian = 0;
    double intensity = 0;

    double ax_dipole = 0;
    double ay_dipole = 0;
    double az_dipole = 0;
    
    double r = sqrt(pow(x,2) + pow(y,2));
    double I0 = (2*P)/(pi*pow(w0,2));

    // out of bounds check
    if (r > (r_check + slope*z)){
        statusi = 3;
    } 
    
    // a_dipole acceleration update
    if (z > 0) {
        w = w0*sqrt(1 + pow((z/zR),2));
        gaussian = exp(-(2*(pow(x,2) + pow(y,2)))/(pow(w,2)));
        intensity = I0*pow((w0/w),2)*gaussian;
        ax_dipole = -4*x*alpha*intensity/(m*pow(w,2));
        ay_dipole = -4*y*alpha*intensity/(m*pow(w,2));
        az_dipole = alpha*intensity*(4*(z/(pow(zR,2)))*pow((w0/w),4)*((pow(x,2) + pow(y,2))/pow(w0,2)) - (2*z/(pow(zR,2)))*pow((w0/w),2))/m;

    }

    else {

        w = w0;
        gaussian = exp(-(2*(pow(x,2) + pow(y,2)))/(pow(w,2)));
        intensity = I0*pow((w0/w),2)*gaussian;
        ax_dipole = -4*x*alpha*intensity/(m*pow(w,2));
        ay_dipole = -4*y*alpha*intensity/(m*pow(w,2));
        az_dipole = 0;

        // inside hcpcf
        if (r < r_hcpcf){

            // stop condition, loaded
            if (z < -100){
                statusi = 1;
            }

        }

        // collided, not loaded
        else if (r > r_hcpcf){

            statusi = 2;

        }

    }

    ax = ax_dipole;
    ay = ay_dipole;
    az = az_dipole - g;
    OPE = -alpha*intensity; 


}


// velocity-verlet updater
__global__ void verlet_integrator(struct Particles atoms[])
{
   
    double KE = 0;
    double GPE = 0;
    double OPE = 0;
    double KE_kick = 0; 

    double intensity = 0;
    double p_scattering = 0;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int particle_id = index; particle_id < N; particle_id += stride){

        for (int T = 1; atoms[particle_id].statusi == 0 && T < T_max; T += 1){ // atoms[particle_id].p[2]>0;  T < 50000
        
            atoms[particle_id].v[0] = atoms[particle_id].v[0] + 0.5*atoms[particle_id].a[0]*dt;
            atoms[particle_id].v[1] = atoms[particle_id].v[1] + 0.5*atoms[particle_id].a[1]*dt;
            atoms[particle_id].v[2] = atoms[particle_id].v[2] + 0.5*atoms[particle_id].a[2]*dt;

            atoms[particle_id].p[0] = atoms[particle_id].p[0] + atoms[particle_id].v[0]*dt;
            atoms[particle_id].p[1] = atoms[particle_id].p[1] + atoms[particle_id].v[1]*dt;
            atoms[particle_id].p[2] = atoms[particle_id].p[2] + atoms[particle_id].v[2]*dt;

            a_dipole(atoms[particle_id].p[0], atoms[particle_id].p[1], atoms[particle_id].p[2], atoms[particle_id].a[0], atoms[particle_id].a[1], atoms[particle_id].a[2], atoms[particle_id].statusi, OPE);

            atoms[particle_id].v[0] = atoms[particle_id].v[0] + 0.5*atoms[particle_id].a[0]*dt;
            atoms[particle_id].v[1] = atoms[particle_id].v[1] + 0.5*atoms[particle_id].a[1]*dt;
            atoms[particle_id].v[2] = atoms[particle_id].v[2] + 0.5*atoms[particle_id].a[2]*dt;

            KE = 0.5*m*(pow(atoms[particle_id].v[0],2) + pow(atoms[particle_id].v[1],2) + pow(atoms[particle_id].v[2],2));

            // scattering kick: ad-hoc implementation
            curandState state_p;
            curand_init((unsigned long long)clock() + index, 0, 0, &state_p);
            p_scattering = curand_uniform_double(&state_p);
            intensity = OPE/(-1*alpha);
            
            if (p_scattering < beta*intensity*dt){

                atoms[particle_id].N_scattering = atoms[particle_id].N_scattering + 1; 

                curandState state_phi;
                curandState state_theta;
                curand_init((unsigned long long)clock() + index, 0, 0, &state_phi);    
                curand_init((unsigned long long)clock() + index, 0, 0, &state_theta);
                double phi = 2*pi*curand_uniform_double(&state_phi);
                double theta = pi*curand_uniform_double(&state_theta);

                atoms[particle_id].v[0] = atoms[particle_id].v[0] + spont_v_factor*cos(phi)*sin(theta);
                atoms[particle_id].v[1] = atoms[particle_id].v[1] + spont_v_factor*sin(phi)*sin(theta);
                atoms[particle_id].v[2] = atoms[particle_id].v[2] + abs_v_factor + spont_v_factor*cos(theta);
                KE_kick = KE - 0.5*m*(pow(atoms[particle_id].v[0],2) + pow(atoms[particle_id].v[1],2) + pow(atoms[particle_id].v[2],2));

            }

            // energy updater
            KE = 0.5*m*(pow(atoms[particle_id].v[0],2) + pow(atoms[particle_id].v[1],2) + pow(atoms[particle_id].v[2],2));
            GPE = m*g*atoms[particle_id].p[2]; 
            atoms[particle_id].energy_current = (KE + GPE + OPE);
            atoms[particle_id].energy_kick_tab = atoms[particle_id].energy_kick_tab + (KE_kick);

            // timer
            atoms[particle_id].t = atoms[particle_id].t + dt; 
            KE_kick = 0;

        }
   
    }

}

void initialize_MB_cloud(struct Particles atoms[]){

    // initializing MB cloud functions
    double *phi, *costheta, *u, *therm; 
    double theta, r;
      
    phi = (double *) malloc(N*sizeof(double));
    costheta = (double *) malloc(N*sizeof(double));
    u = (double *) malloc(N*sizeof(double));
    therm = (double *) malloc(3*N*sizeof(double));
          
    // random number generation   
    double *d_phi, *d_costheta, *d_u, *d_therm;
    cudaMalloc(&d_phi, N*sizeof(double));
    cudaMalloc(&d_costheta, N*sizeof(double));
    cudaMalloc(&d_u, N*sizeof(double));
    cudaMalloc(&d_therm, 3*N*sizeof(double));

    // unsigned int seeder, seeder1, seeder2, seeder3;
    // seeder = 1234ULL;
    // seeder1 = 1234ULL;
    // seeder2 = 1234ULL;
    // seeder3 = 1234ULL;

    time_t seeder;
    time(&seeder);
    srand((unsigned int) seeder);
   
    // generates N random numbers between 0 and 1
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, seeder);
    curandGenerateUniformDouble(gen, d_phi, N);
    cudaMemcpy(phi, d_phi, N*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen);
    cudaFree(d_phi);
    
    time_t seeder1;
    time(&seeder1);
    srand((unsigned int) seeder1);
    
    curandGenerator_t gen1;
    curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen1, seeder1);
    curandGenerateUniformDouble(gen1, d_costheta, N);
    cudaMemcpy(costheta, d_costheta, N*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen1);
    cudaFree(d_costheta);

   
    time_t seeder2;
    time(&seeder2);
    srand((unsigned int) seeder2);
      
    curandGenerator_t gen2;
    curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen2, seeder2);
    curandGenerateUniformDouble(gen2, d_u, N);
    cudaMemcpy(u, d_u, N*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen2);
    cudaFree(d_u);
   
    time_t seeder3;
    time(&seeder3);
    srand((unsigned int) seeder3);
      
    // normal distribution to sample thermal velocities 
    curandGenerator_t gen3;
    curandCreateGenerator(&gen3, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen3, seeder3);
    curandGenerateNormalDouble(gen3, d_therm, 3*N, 0, 1);
    cudaMemcpy(therm, d_therm, 3*N*sizeof(double), cudaMemcpyDeviceToHost);
    curandDestroyGenerator(gen3);
    cudaFree(d_therm);

    // initializing (p, v, a, t, status) array values for atoms
    for (int i = 0; i < N; i++){

        phi[i] = phi[i]*2*pi;
        costheta[i] = costheta[i]*2 - 1;

        theta = acos(costheta[i]);
        r = R*cbrt(u[i]);

        atoms[i].p[0] = cx + r*sin(theta)*cos(phi[i]);
        atoms[i].p[1] = cy + r*sin(theta)*sin(phi[i]);
        atoms[i].p[2] = cz + r*cos(theta);

        atoms[i].v[0] = sqrt(kB*Temp/m)*therm[i];
        atoms[i].v[1] = sqrt(kB*Temp/m)*therm[i + N - 1];
        atoms[i].v[2] = sqrt(kB*Temp/m)*therm[i + 2*N - 1];

        atoms[i].a[0] = 0;
        atoms[i].a[1] = 0;
        atoms[i].a[2] = -g;

        atoms[i].N_scattering = 0; 

        double KE = 0.5*m*(pow(atoms[i].v[0],2) + pow(atoms[i].v[1],2) + pow(atoms[i].v[2],2));
        double GE = m*g*atoms[i].p[2];

        double x = atoms[i].p[0];
        double y = atoms[i].p[1];
        double z = atoms[i].p[2];

        // calculating dipole field and potneial 

        double I0 = (2*P)/(pi*pow(w0,2));

        double w = w0*sqrt(1 + pow((z/zR),2));
        double gaussian = exp(-(2*(pow(x,2) + pow(y,2)))/(pow(w,2)));
        double intensity = I0*pow((w0/w),2)*gaussian;
        double OPE = -alpha*intensity;

        atoms[i].energy_current = (KE + GE + OPE);
        atoms[i].energy_initial = atoms[i].energy_current;
        double r = sqrt(pow(atoms[i].p[0], 2) + pow(atoms[i].p[1], 2)); // has to be sqrt
    
        if (r > (r_check + slope*atoms[i].p[2])){
            atoms[i].statusi = 3;
        } 

    }

}




int main(void){

    // AoS allocating memory
    struct Particles *Atoms = (struct Particles*)malloc(N*sizeof(struct Particles));

    // initializing atoms
    initialize_MB_cloud(Atoms);

    //************************************************//
    printf("----------------------------\n");
    printf("Before Evolution:\n");
    printf("px = %lf\n", Atoms[0].p[0]);
    printf("py = %lf\n", Atoms[0].p[1]);
    printf("pz = %lf\n", Atoms[0].p[2]);
           
    printf("vx = %lf\n", Atoms[0].v[0]);
    printf("vy = %lf\n", Atoms[0].v[1]);
    printf("vz = %lf\n", Atoms[0].v[2]);

    
    std::cout.precision(dbl::digits10);
    std::cout << "az (precision) " << Atoms[0].a[2] << std::endl;
    printf("----------------------------\n");
    //************************************************//

    printf("Memory Allocation Begin\n");

    // device: 1D arrays for position, velocity and acceleration
    Particles *dev_Atoms;

    // device: allocation of  memory 
    cudaMalloc(&dev_Atoms, N*sizeof(Particles)); 

    // device: copying initial values 
    cudaMemcpy(dev_Atoms, Atoms, N*sizeof(Particles), cudaMemcpyHostToDevice);
       

    printf("Memory Allocation End \n");

    // kernel initialization
    // code reference:  https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels

    int blockSize;      // The launch configurator returned block size 
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    int gridSize;       // The actual grid size needed, based on input size 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, verlet_integrator, 0, N); 

    // Round up according to array size 
    gridSize = (N + blockSize - 1) / blockSize; 

    printf("Grid Size (GPU, AoS) = %d\n", gridSize); 
    printf("Block Size (GPU, AoS) = %d\n", blockSize); 


    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);


    // testing timing of one verlet update of N particles
    cudaEventRecord(cuda_start);
    verlet_integrator<<<gridSize, blockSize>>>(dev_Atoms);
    cudaEventRecord(cuda_stop);
    
    cudaEventSynchronize(cuda_stop);
    float cuda_diffs = 0;
    cudaEventElapsedTime(&cuda_diffs, cuda_start, cuda_stop);
    cuda_diffs = cuda_diffs/1000.0;
    printf("GPU Time for Verlet integration (GPU, AoS) = %lf\n", cuda_diffs);

    
    // device: copying initial values 
    cudaMemcpy(Atoms, dev_Atoms, N*sizeof(Particles), cudaMemcpyDeviceToHost);

    //************************************************//
    printf("----------------------------\n");
    printf("After Evolution:\n");
    printf("px = %lf\n", Atoms[0].p[0]);
    printf("py = %lf\n", Atoms[0].p[1]);
    printf("pz = %lf\n", Atoms[0].p[2]);
       
    printf("vx = %lf\n", Atoms[0].v[0]);
    printf("vy = %lf\n", Atoms[0].v[1]);
    printf("vz = %lf\n", Atoms[0].v[2]);
 
    std::cout.precision(dbl::digits10);
    std::cout << "az (precision) " << Atoms[0].a[2] << std::endl;
    printf("----------------------------\n");
    //************************************************//
    
    int completed = 0; 
    int trapped = 0;
    int escaped = 0; 
    int running = 0;

    int unstable_integration_counter = 0; 

    double *fractional_energy_change;
    fractional_energy_change = (double *) malloc(N*sizeof(double));

    int fractional_energy_change_trapped = 0; 

    double N_scattering_average_total = 0;
    double N_scattering_average_loaded = 0;

    double loading_efficiency; 
    for (int i = 0; i < N; i++){
        if (Atoms[i].statusi == 1 || Atoms[i].statusi == 2) {
            completed ++; 
        }

        if (Atoms[i].statusi == 1){
            trapped++;
            fractional_energy_change[i] = abs(Atoms[i].energy_current  + Atoms[i].energy_kick_tab - Atoms[i].energy_initial)/(Atoms[i].energy_initial);
            if (fractional_energy_change[i] > energy_error_check){
                fractional_energy_change_trapped++;
            }
            N_scattering_average_loaded = N_scattering_average_loaded + Atoms[i].N_scattering;
        }
        
        if (Atoms[i].statusi == 3){
            escaped++;
        }

        if (Atoms[i].statusi == 0){
            running++;
        }


        fractional_energy_change[i] = abs(Atoms[i].energy_current + Atoms[i].energy_kick_tab - Atoms[i].energy_initial)/(Atoms[i].energy_initial);
        if (fractional_energy_change[i] > energy_error_check){
            unstable_integration_counter++;
            // std::cout << "Energy instability (> 0.05): " << Atoms_GPU[i].p[0] << ", " << Atoms_GPU[i].p[1] << ", " << Atoms_GPU[i].p[2] << ", " << Atoms_GPU[i].statusi << std::endl;

        }
     

        N_scattering_average_total = N_scattering_average_total + Atoms[i].N_scattering;

    }
    
    N_scattering_average_total = N_scattering_average_total/N; 
    N_scattering_average_loaded = N_scattering_average_loaded/trapped;

    loading_efficiency = (double)trapped/N;

    printf("Total Number of Atoms = %d\n", N);
    printf("Accounted Number of Atoms = %d\n", completed + escaped + running);
    printf("Completed Atom Runs = %d\n", completed);
    printf("Number of Escaped Atoms = %d\n", escaped);
    printf("Number of Trapped Atoms = %d\n", trapped);
    printf("Number of Running Atoms = %d\n", running);
    printf("Loading Efficiency (1e-4) = %lf\n", loading_efficiency*1e4);


    double fractional_energy_change_min = fractional_energy_change[0];                
    for (int j = 0; j < N; j++){
        if (fractional_energy_change[j] < fractional_energy_change_min){
            fractional_energy_change_min = fractional_energy_change[j];
            if (fractional_energy_change_min == 0){
                std::cout << Atoms[j].p[0] << ", " << Atoms[j].p[1] << ", " << Atoms[j].p[2] << ", " << Atoms[j].statusi << std::endl;
            }
        }
    }


    double fractional_energy_change_max = fractional_energy_change[0];                
    for (int j = 0; j < N; j++){
        if (fractional_energy_change[j] > fractional_energy_change_max){
            fractional_energy_change_max = fractional_energy_change[j];
        }
    }

    printf("Total number of fractional errors in energy : %d\n", unstable_integration_counter);
    printf("Number of fractional errors that are loaded: %d\n", fractional_energy_change_trapped);
    printf("----------------------------\n");
    printf("Fractional energy change max: %lf\n", fractional_energy_change_max);
    printf("Fractional energy change min: %lf\n", fractional_energy_change_min);
    std::cout << "Fractional energy change min (precision) " << fractional_energy_change_min << std::endl;
    printf("----------------------------\n");

    printf("----------------------------\n");
    printf("Fractional energy change max: %lf\n", fractional_energy_change_max);
    printf("----------------------------\n");

    printf("Average No. of Scattering Events for Loaded Atoms = %lf\n", N_scattering_average_loaded);

    //************************************************/
    // clearing memory    
    cudaFree(dev_Atoms);
    free(Atoms);

    cudaError_t cudaResult;
    cudaResult = cudaGetLastError();
    if (cudaResult != cudaSuccess) {
        printf("In what house, shall thy find solace? ");

    }

}
