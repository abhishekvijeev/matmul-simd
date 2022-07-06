#include "bl_config.h"
#include "bl_dgemm.h"
#include "bl_dgemm_kernel.h"

#include <immintrin.h>

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k, // Kc
		           int    m, // Mr
                   int    n, // Nr
                   const double * restrict a,
                   const double * restrict b,
                   double *c,
                   unsigned long long lda,
                   unsigned long long ldb,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

   
    for ( l = 0; l < k; ++l )
    {
        for ( i = 0; i < m; ++i )
        {
            for ( j = 0; j < n; ++j )
            {
                // ldc is used here because a[] and b[] are not packed by the
                // starter code
                // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
                //
                c( i, j, ldc ) += a( l, i, lda) * b( l, j, ldb );
                //   size_t c_index = (i)*(ldc) + (j);
                //   size_t a_index = (l)*(4) + (i);
                //   size_t b_index = (l)*(4) + (j);
                //   printf("c[%lu] += a[%lu] * b[%lu]\n", c_index, a_index, b_index);
            }
        }
    }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//

// AVX based microkernel
void bl_dgemm_avx_4x4_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc) );
    register __m256d c20_c21_c22_c23 = _mm256_loadu_pd( c + (2 * ldc) );
    register __m256d c30_c31_c32_c33 = _mm256_loadu_pd( c + (3 * ldc) );

    for (int l = 0; l < kc; l++){
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda) );
        register __m256d a1x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        register __m256d a2x = _mm256_broadcast_sd ( a + (l * lda) + 2);
        register __m256d a3x = _mm256_broadcast_sd ( a + (l * lda) + 3);
        register __m256d b_vec = _mm256_loadu_pd(b + (l * ldb));
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b_vec, c00_c01_c02_c03);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b_vec, c10_c11_c12_c13);
        c20_c21_c22_c23 = _mm256_fmadd_pd (a2x, b_vec, c20_c21_c22_c23);
        c30_c31_c32_c33 = _mm256_fmadd_pd (a3x, b_vec, c30_c31_c32_c33);
    }

    _mm256_storeu_pd(c, c00_c01_c02_c03);
    _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
    _mm256_storeu_pd(c + (2 * ldc), c20_c21_c22_c23);
    _mm256_storeu_pd(c + (3 * ldc), c30_c31_c32_c33);

}


void bl_dgemm_avx_7x4_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc) );
    register __m256d c20_c21_c22_c23 = _mm256_loadu_pd( c + (2 * ldc) );
    register __m256d c30_c31_c32_c33 = _mm256_loadu_pd( c + (3 * ldc) );
    register __m256d c40_c41_c42_c43 = _mm256_loadu_pd( c + (4 * ldc));
    register __m256d c50_c51_c52_c53 = _mm256_loadu_pd( c + (5 * ldc) );
    register __m256d c60_c61_c62_c63 = _mm256_loadu_pd( c + (6 * ldc) );

    for (int l = 0; l < kc; l++){
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda) );
        register __m256d a1x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        register __m256d a2x = _mm256_broadcast_sd ( a + (l * lda) + 2);
        register __m256d a3x = _mm256_broadcast_sd ( a + (l * lda) + 3);
        register __m256d a4x = _mm256_broadcast_sd ( a + (l * lda) + 4);
        register __m256d a5x = _mm256_broadcast_sd ( a + (l * lda) + 5);
        register __m256d a6x = _mm256_broadcast_sd ( a + (l * lda) + 6);   
        register __m256d b_vec = _mm256_loadu_pd(b + (l * ldb));
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b_vec, c00_c01_c02_c03);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b_vec, c10_c11_c12_c13);
        c20_c21_c22_c23 = _mm256_fmadd_pd (a2x, b_vec, c20_c21_c22_c23);
        c30_c31_c32_c33 = _mm256_fmadd_pd (a3x, b_vec, c30_c31_c32_c33);
        c40_c41_c42_c43 = _mm256_fmadd_pd (a4x, b_vec, c40_c41_c42_c43);
        c50_c51_c52_c53 = _mm256_fmadd_pd (a5x, b_vec, c50_c51_c52_c53);
        c60_c61_c62_c63 = _mm256_fmadd_pd (a6x, b_vec, c60_c61_c62_c63);
    }

    _mm256_storeu_pd(c, c00_c01_c02_c03);
    _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
    _mm256_storeu_pd(c + (2 * ldc), c20_c21_c22_c23);
    _mm256_storeu_pd(c + (3 * ldc), c30_c31_c32_c33);
    _mm256_storeu_pd(c + (4 * ldc), c40_c41_c42_c43);
    _mm256_storeu_pd(c + (5 * ldc), c50_c51_c52_c53);
    _mm256_storeu_pd(c + (6 * ldc), c60_c61_c62_c63);
}


void bl_dgemm_avx_4x8_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c04_c05_c06_c07 = _mm256_loadu_pd( c + 4);
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc) );
    register __m256d c14_c15_c16_c17 = _mm256_loadu_pd( c + (1 * ldc) + 4);
    register __m256d c20_c21_c22_c23 = _mm256_loadu_pd( c + (2 * ldc) );
    register __m256d c24_c25_c26_c27 = _mm256_loadu_pd( c + (2 * ldc) + 4);
    register __m256d c30_c31_c32_c33 = _mm256_loadu_pd( c + (3 * ldc) );
    register __m256d c34_c35_c36_c37 = _mm256_loadu_pd( c + (3 * ldc) + 4);
    
    

    for (int l = 0; l < kc; l++){
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda) );
        register __m256d a1x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        register __m256d a2x = _mm256_broadcast_sd ( a + (l * lda) + 2);
        register __m256d a3x = _mm256_broadcast_sd ( a + (l * lda) + 3);
        register __m256d b0_vec = _mm256_loadu_pd(b + (l * ldb));
        register __m256d b1_vec = _mm256_loadu_pd(b + (l * ldb) + 4);
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b0_vec, c00_c01_c02_c03);
        c04_c05_c06_c07 = _mm256_fmadd_pd (a0x, b1_vec, c04_c05_c06_c07);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b0_vec, c10_c11_c12_c13);
        c14_c15_c16_c17 = _mm256_fmadd_pd (a1x, b1_vec, c14_c15_c16_c17);
        c20_c21_c22_c23 = _mm256_fmadd_pd (a2x, b0_vec, c20_c21_c22_c23);
        c24_c25_c26_c27 = _mm256_fmadd_pd (a2x, b1_vec, c24_c25_c26_c27);
        c30_c31_c32_c33 = _mm256_fmadd_pd (a3x, b0_vec, c30_c31_c32_c33);
        c34_c35_c36_c37 = _mm256_fmadd_pd (a3x, b1_vec, c34_c35_c36_c37);
    }

    _mm256_storeu_pd(c, c00_c01_c02_c03);
    _mm256_storeu_pd(c + 4, c04_c05_c06_c07);
    _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
    _mm256_storeu_pd(c + (1 * ldc) + 4, c14_c15_c16_c17);
    _mm256_storeu_pd(c + (2 * ldc), c20_c21_c22_c23);
    _mm256_storeu_pd(c + (2 * ldc) + 4, c24_c25_c26_c27);
    _mm256_storeu_pd(c + (3 * ldc), c30_c31_c32_c33);
    _mm256_storeu_pd(c + (3 * ldc) + 4, c34_c35_c36_c37);
}

void bl_dgemm_avx_2x16_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c04_c05_c06_c07 = _mm256_loadu_pd( c + 4);
    register __m256d c08_c09_c10_c11 = _mm256_loadu_pd( c + 8);
    register __m256d c012_c013_c014_c015 = _mm256_loadu_pd( c + 12);
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc));
    register __m256d c14_c15_c16_c17 = _mm256_loadu_pd( c + (1 * ldc) + 4);
    register __m256d c18_c19_c110_c111 = _mm256_loadu_pd( c + (1 * ldc) + 8);
    register __m256d c112_c113_c114_c115 = _mm256_loadu_pd( c + (1 * ldc) + 12);

    for (int l = 0; l < kc; l++){
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda));
        register __m256d a1x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        register __m256d b0_vec = _mm256_loadu_pd(b + (l * ldb));
        register __m256d b1_vec = _mm256_loadu_pd(b + (l * ldb) + 4);
        register __m256d b2_vec = _mm256_loadu_pd(b + (l * ldb) + 8);
        register __m256d b3_vec = _mm256_loadu_pd(b + (l * ldb) + 12);
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b0_vec, c00_c01_c02_c03);
        c04_c05_c06_c07 = _mm256_fmadd_pd (a0x, b1_vec, c04_c05_c06_c07);
        c08_c09_c10_c11 = _mm256_fmadd_pd (a0x, b2_vec, c08_c09_c10_c11);
        c012_c013_c014_c015 = _mm256_fmadd_pd (a0x, b3_vec, c012_c013_c014_c015);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b0_vec, c10_c11_c12_c13);
        c14_c15_c16_c17 = _mm256_fmadd_pd (a1x, b1_vec, c14_c15_c16_c17);
        c18_c19_c110_c111 = _mm256_fmadd_pd (a1x, b2_vec, c18_c19_c110_c111);
        c112_c113_c114_c115 = _mm256_fmadd_pd (a1x, b3_vec, c112_c113_c114_c115);
    }

    _mm256_storeu_pd(c, c00_c01_c02_c03);
    _mm256_storeu_pd(c + 4, c04_c05_c06_c07);
    _mm256_storeu_pd(c + 8, c08_c09_c10_c11);
    _mm256_storeu_pd(c + 12, c012_c013_c014_c015);
    _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
    _mm256_storeu_pd(c + (1 * ldc) + 4, c14_c15_c16_c17);
    _mm256_storeu_pd(c + (1 * ldc) + 8, c18_c19_c110_c111);
    _mm256_storeu_pd(c + (1 * ldc) + 12, c112_c113_c114_c115);
}

void bl_dgemm_avx_2x8_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c04_c05_c06_c07 = _mm256_loadu_pd( c + 4);
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc) );
    register __m256d c14_c15_c16_c17 = _mm256_loadu_pd( c + (1 * ldc) + 4);  
    

    for (int l = 0; l < kc; l++){
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda) );
        register __m256d a1x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        register __m256d b0_vec = _mm256_loadu_pd(b + (l * ldb));
        register __m256d b1_vec = _mm256_loadu_pd(b + (l * ldb) + 4);
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b0_vec, c00_c01_c02_c03);
        c04_c05_c06_c07 = _mm256_fmadd_pd (a0x, b1_vec, c04_c05_c06_c07);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b0_vec, c10_c11_c12_c13);
        c14_c15_c16_c17 = _mm256_fmadd_pd (a1x, b1_vec, c14_c15_c16_c17);
    }

    _mm256_storeu_pd(c, c00_c01_c02_c03);
    _mm256_storeu_pd(c + 4, c04_c05_c06_c07);
    _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
    _mm256_storeu_pd(c + (1 * ldc) + 4, c14_c15_c16_c17);
}


void bl_dgemm_avx_2x4_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc) );


    for (int l = 0; l < kc; l++) {
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda) );
        register __m256d a1x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        register __m256d b0_vec = _mm256_loadu_pd(b + (l * ldb));
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b0_vec, c00_c01_c02_c03);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b0_vec, c10_c11_c12_c13);
    }

    _mm256_storeu_pd(c, c00_c01_c02_c03);
    _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
}

void bl_dgemm_avx_4x12_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c04_c05_c06_c07 = _mm256_loadu_pd( c + 4);
    register __m256d c08_c09_c10_c11 = _mm256_loadu_pd( c + 8);
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc) );
    register __m256d c14_c15_c16_c17 = _mm256_loadu_pd( c + (1 * ldc) + 4);
    register __m256d c18_c19_c110_c111 = _mm256_loadu_pd( c + (1 * ldc) + 8);
    register __m256d c20_c21_c22_c23 = _mm256_loadu_pd( c + (2 * ldc) );
    register __m256d c24_c25_c26_c27 = _mm256_loadu_pd( c + (2 * ldc) + 4);
    register __m256d c28_c29_c210_c211 = _mm256_loadu_pd( c + (2 * ldc) + 8);
    register __m256d c30_c31_c32_c33 = _mm256_loadu_pd( c + (3 * ldc) );
    register __m256d c34_c35_c36_c37 = _mm256_loadu_pd( c + (3 * ldc) + 4);
    register __m256d c38_c39_c310_c311 = _mm256_loadu_pd( c + (3 * ldc) + 8);

    
    for (int l = 0; l < kc; l++){
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda) );
        register __m256d b0_vec = _mm256_loadu_pd(b + (l * ldb));
        register __m256d b1_vec = _mm256_loadu_pd(b + (l * ldb) + 4);
        register __m256d b2_vec = _mm256_loadu_pd(b + (l * ldb) + 8);
        
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b0_vec, c00_c01_c02_c03);
        c04_c05_c06_c07 = _mm256_fmadd_pd (a0x, b1_vec, c04_c05_c06_c07);
        c08_c09_c10_c11 = _mm256_fmadd_pd (a0x, b2_vec, c08_c09_c10_c11);
        
        a0x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a0x, b0_vec, c10_c11_c12_c13);
        c14_c15_c16_c17 = _mm256_fmadd_pd (a0x, b1_vec, c14_c15_c16_c17);
        c18_c19_c110_c111 = _mm256_fmadd_pd (a0x, b2_vec, c18_c19_c110_c111);

        a0x = _mm256_broadcast_sd ( a + (l * lda) + 2);
        c20_c21_c22_c23 = _mm256_fmadd_pd (a0x, b0_vec, c20_c21_c22_c23);
        c24_c25_c26_c27 = _mm256_fmadd_pd (a0x, b1_vec, c24_c25_c26_c27);
        c28_c29_c210_c211 = _mm256_fmadd_pd (a0x, b2_vec, c28_c29_c210_c211);

        a0x = _mm256_broadcast_sd ( a + (l * lda) + 3);
        c30_c31_c32_c33 = _mm256_fmadd_pd (a0x, b0_vec, c30_c31_c32_c33);
        c34_c35_c36_c37 = _mm256_fmadd_pd (a0x, b1_vec, c34_c35_c36_c37);
        c38_c39_c310_c311 = _mm256_fmadd_pd (a0x, b2_vec, c38_c39_c310_c311);
    }

    _mm256_storeu_pd(c, c00_c01_c02_c03);
    _mm256_storeu_pd(c + 4, c04_c05_c06_c07);
    _mm256_storeu_pd(c + 8, c08_c09_c10_c11);
    _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
    _mm256_storeu_pd(c + (1 * ldc) + 4, c14_c15_c16_c17);
    _mm256_storeu_pd(c + (1 * ldc) + 8, c18_c19_c110_c111);
    _mm256_storeu_pd(c + (2 * ldc), c20_c21_c22_c23);
    _mm256_storeu_pd(c + (2 * ldc) + 4, c24_c25_c26_c27);
    _mm256_storeu_pd(c + (2 * ldc) + 8, c28_c29_c210_c211);
    _mm256_storeu_pd(c + (3 * ldc), c30_c31_c32_c33);
    _mm256_storeu_pd(c + (3 * ldc) + 4, c34_c35_c36_c37);
    _mm256_storeu_pd(c + (3 * ldc) + 8, c38_c39_c310_c311);
}


void bl_dgemm_avx_2x24_ukr(  int    kc, // Kc
                            const double * restrict a,
                            const double * restrict b,
                            double *c,
                            unsigned long long lda,
                            unsigned long long ldb,
                            unsigned long long ldc,
                            aux_t* data )
{
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd( c );
    register __m256d c04_c05_c06_c07 = _mm256_loadu_pd( c + 4 );
    register __m256d c08_c09_c010_c011 = _mm256_loadu_pd( c + 8 );
    register __m256d c012_c013_c014_c015 = _mm256_loadu_pd( c + 12 );
    register __m256d c016_c017_c018_c019 = _mm256_loadu_pd( c + 16 );
    register __m256d c020_c021_c022_c023 = _mm256_loadu_pd( c + 20 );
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd( c + (1 * ldc) );
    register __m256d c14_c15_c16_c17 = _mm256_loadu_pd( c + (1 * ldc) + 4 );
    register __m256d c18_c19_c110_c111 = _mm256_loadu_pd( c + (1 * ldc) + 8 );
    register __m256d c112_c113_c114_c115 = _mm256_loadu_pd( c + (1 * ldc) + 12 );
    register __m256d c116_c117_c118_c119 = _mm256_loadu_pd( c + (1 * ldc) + 16 );
    register __m256d c120_c121_c122_c123 = _mm256_loadu_pd( c + (1 * ldc) + 20 );
    
    
    
    for (int l = 0; l < kc; l++){
        register __m256d a0x = _mm256_broadcast_sd ( a + (l * lda) );
        register __m256d a1x = _mm256_broadcast_sd ( a + (l * lda) + 1);
        register __m256d b0_vec = _mm256_loadu_pd(b + (l * ldb));
        register __m256d b1_vec = _mm256_loadu_pd(b + (l * ldb) + 4);
        
        c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b0_vec, c00_c01_c02_c03);
        c04_c05_c06_c07 = _mm256_fmadd_pd (a0x, b1_vec, c04_c05_c06_c07);
        c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b0_vec, c10_c11_c12_c13);
        c14_c15_c16_c17 = _mm256_fmadd_pd (a1x, b1_vec, c14_c15_c16_c17);
        
        b0_vec = _mm256_loadu_pd(b + (l * ldb) +8);
        b1_vec = _mm256_loadu_pd(b + (l * ldb) + 12);
        c08_c09_c010_c011 = _mm256_fmadd_pd (a0x, b0_vec, c08_c09_c010_c011);
        c012_c013_c014_c015 = _mm256_fmadd_pd (a0x, b1_vec, c012_c013_c014_c015);
        c18_c19_c110_c111 = _mm256_fmadd_pd (a1x, b0_vec, c18_c19_c110_c111);
        c112_c113_c114_c115 = _mm256_fmadd_pd (a1x, b1_vec, c112_c113_c114_c115);

        b0_vec = _mm256_loadu_pd(b + (l * ldb) +16);
        b1_vec = _mm256_loadu_pd(b + (l * ldb) + 20);
        c016_c017_c018_c019 = _mm256_fmadd_pd (a0x, b0_vec, c016_c017_c018_c019);
        c020_c021_c022_c023 = _mm256_fmadd_pd (a0x, b1_vec, c020_c021_c022_c023);
        c116_c117_c118_c119 = _mm256_fmadd_pd (a1x, b0_vec, c116_c117_c118_c119);
        c120_c121_c122_c123 = _mm256_fmadd_pd (a1x, b1_vec, c120_c121_c122_c123);
    }

        _mm256_storeu_pd(c, c00_c01_c02_c03);
        _mm256_storeu_pd(c + 4, c04_c05_c06_c07);
        _mm256_storeu_pd(c + 8, c08_c09_c010_c011);
        _mm256_storeu_pd(c + 12, c012_c013_c014_c015);
        _mm256_storeu_pd(c + 16, c016_c017_c018_c019);
        _mm256_storeu_pd(c + 20, c020_c021_c022_c023);
        _mm256_storeu_pd(c + (1 * ldc), c10_c11_c12_c13);
        _mm256_storeu_pd(c + (1 * ldc) + 4, c14_c15_c16_c17);
        _mm256_storeu_pd(c + (1 * ldc) + 8, c18_c19_c110_c111);
        _mm256_storeu_pd(c + (1 * ldc) + 12, c112_c113_c114_c115);
        _mm256_storeu_pd(c + (1 * ldc) + 16, c116_c117_c118_c119);
        _mm256_storeu_pd(c + (1 * ldc) + 20, c120_c121_c122_c123);
}


void bl_dgemm_simd_ukr( int    k, // Kc
                        int    m, // Mr
                        int    n, // Nr
                        const double * restrict a,
                        const double * restrict b,
                        double *c,
                        unsigned long long ldc,
                        aux_t* data )
{
     int l, j, i;
    
    unsigned long long lda = m;
    unsigned long long ldb = n;


    for ( l = 0; l < k; l += 64 ) {
        int l_min = min(64, (k - l));
        for ( i = 0; i < m; i += 4 ) {
            int i_min = min(4, (m - i));
            for ( j = 0; j < n; j += 12 ) {
                int j_min = min(12, (n - j));

                // if (i_min == 2 && j_min == 24) {
                //     bl_dgemm_avx_2x24_ukr(l_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                // }
                if (i_min == 4 && j_min == 12) {
                    bl_dgemm_avx_4x12_ukr(l_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                }
                else if (i_min == 4 && j_min == 8) {
                    bl_dgemm_avx_4x8_ukr(l_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                }
                // else if (i_min == 2 && j_min == 16) {
                //     bl_dgemm_avx_2x16_ukr(l_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                //     //j += 4;
                //  }
                else if (i_min == 4 && j_min == 4) {
                    bl_dgemm_avx_4x4_ukr(l_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                }
                else if (i_min == 2 && j_min == 8) {
                    bl_dgemm_avx_2x8_ukr(l_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                }
                else if (i_min == 2 && j_min == 4) {
                    bl_dgemm_avx_2x4_ukr(l_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                }
                else {
                    bl_dgemm_ukr(l_min, i_min, j_min, &a[ (l*m) + i ], &b[ (l*n) + j ], &c[ (i*ldc) + j], lda, ldb, ldc, data);
                }
            }
        }
    }

}



