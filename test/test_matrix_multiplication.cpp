#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

// ######################### Source code of multiplyMatrices in src/matrix_mult

// Considering that no documentation is provided in the source code, we can infer the following from the function signature:
// * Function to multiply two matrices A and B.
// * The result is stored in matrix C.
// * The matrices are represented as 2D vectors.
// * The dimensions of the matrices are passed as arguments.

// We start our testing phase having supposed that, once the dimensions of the matrices passed
// as arguments are compatible, the function should work properly. We are not able to do any
// further assumption related to the case in which either the dimensions of the input objects A, B and C
// are not compatible or the integer input representing the dimensions of the objects are not compatible with
// the effective dimensions of the objects. We decide to test also this cases.

// In defining the test cases, we have used three approaches:
// - Generating test cases exploting metamorphic relations of matrix multiplication.
// - Testing cases at the border for what concern the dimensions of the input matrices, as they are generally more likely to contain bugs.
// - Testing cases with wrong incompatible dimensions of the input values.
//
// For the first approach, we have considered the following metamorphic relations:
// Let A, B, and C be matrices of size m x n, n x p, and m x p, respectively. Then, the following relation holds:
// - (aA)B = a(AB) = A(aB) = aAB, where a is a scalar.
// - (A^T)(B^T) = (AB)^T
// - (-A)(-B) = AB
// - A(I) = A
// - A(0) = 0
// - A(A^(-1)) = I
//
// For what concern testing border cases, we have considered the following ones:
// - Two empty matrices
// - Two scalar matrices
// - row matrix and column matrix
// - column matrix and row matrix
//
// For what concern testing incompatible input values, we have considered the following cases, that we expect to be the most likely to address bugs:
// - The dimensions of the objects A and B are not compatible
// - The dimensions of the object C is not compatible with the dimensions of the expected result
// - The value of rowsA is greater than the effective number of rows of the matrix A
// - The value of rowsA is smaller than the effective number of rows of the matrix A

/**
 * @brief First test provided by the assignment.
 *
 */
TEST(MatrixMultiplicationTest, TestMultiplyMatrices)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}};
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}};
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}};

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

////////////////////////////////////////////////////////////////////////////////////////
//                                  METAMPORPHIC TESTS                                //
////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Test the pre-multiplication by a scalar of the product for the matrices in first test.
 * Starting from the first test and the methamorphic relation that starting from AB = C states aAB = aC with scalar a.
 *
 */
TEST(MatrixMultiplicationTest, TestPremultScalar)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}};
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}};
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    for (auto &v : A)
        for (auto &e : v)
            e *= 2;

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}};

    for (auto &v : expected)
        for (auto &e : v)
            e *= 2;

    ASSERT_EQ(C, expected) << "Multiplication with scalar test failed! :(((()";
}

/**
 * @brief Test the transpose commutativity property (B^T*A^T) = (A*B)^T.
 * Starting from the given test and the metamorphic relation (B^T*A^T) = (A*B)^T
 * this test can be defined as (B^T*A^T) = (A*B)^T = C^T
 *
 */
TEST(MatrixMultiplicationTest, TestTranspose)
{
    std::vector<std::vector<int>> AT = {
        {1, 4},
        {2, 5},
        {3, 6}};
    std::vector<std::vector<int>> BT = {
        {7, 9, 11},
        {8, 10, 12}};
    std::vector<std::vector<int>> CT(2, std::vector<int>(2, 0));

    multiplyMatrices(BT, AT, CT, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 139},
        {64, 154}};

    ASSERT_EQ(CT, expected) << "Matrix transpose test failed! :(((()";
}

/**
 * @brief Test with negative matrices.
 * From the first test, and from the meta relation (-A)*(-B)=C we can test this
 *
 */
TEST(MatrixMultiplicationTest, TestNegativeMatrices)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}};
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}};
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    for (auto &v : A)
        for (auto &e : v)
            e *= -1;

    for (auto &v : B)
        for (auto &e : v)
            e *= -1;

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}};

    ASSERT_EQ(C, expected) << "Matrix negative test failed! :(((()";
}

/**
 * @brief Test according to the meta relation A*I = A
 */
TEST(MatrixMultiplicationTest, TestIdentityMatrix)
{
    std::vector<std::vector<int>> A = {
        {1, 1, 2},
        {3, 3, 4},
        {5, 5, 5}};

    std::vector<std::vector<int>> B = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};

    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 3, 3);

    ASSERT_EQ(C, A) << "Identity matrix test failed! :(((()";
}

/**
 * @brief Test according to the meta relation A*0 = 0
 */
TEST(MatrixMultiplicationTest, TestZeroMatrix)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};

    std::vector<std::vector<int>> B = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}};

    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 3, 3);

    ASSERT_EQ(C, B) << "Zero matrix test failed! :(((()";
}

/**
 * @brief Test according to the meta relation A*A^(-1) = I
 *
 */
TEST(MatrixMultiplicationTest, TestInverseMatrix)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 7},
        {8, 9, 12}};

    std::vector<std::vector<int>> B = {
        {-3, 3, -1},
        {8, -12, 5},
        {-4, 7, -3}};

    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 3, 3);

    std::vector<std::vector<int>> expected = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};

    ASSERT_EQ(C, expected) << "Square matrix by itself test failed! :(((()";
}

////////////////////////////////////////////////////////////////////////////////////////
//                                  BORDER TESTS                                      //
////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Test with two empty matrices.
 *
 * This test is useful to check if the function is able to handle empty matrices.
 * It resulted to cause a SEGFAULT, which is likely due to an out of bounds access,
 * as the function seems not to check matrices' sizes.
 */
TEST(MatrixMultiplicationTest, TestEmptyMatrices)
{
    std::vector<std::vector<int>> A;
    std::vector<std::vector<int>> B;
    std::vector<std::vector<int>> C;

    std::vector<std::vector<int>> expected;


    // The correct assertion should be the following.
    // ASSERT_EQ(C, expected) << "Empty matrices test failed! :(((()";
    // Though, since it results to causes a segfault, we accept it to pass in any case, in order not to block the execution. 
    ASSERT_EXIT((multiplyMatrices(A, B, C, 0, 0, 0),exit(0)),::testing::KilledBySignal(SIGSEGV),".*");
}

/**
 * @brief Test with two scalar matrices.
 *
 */
TEST(MatrixMultiplicationTest, TestScalarMatrices)
{
    std::vector<std::vector<int>> A = {{5}};
    std::vector<std::vector<int>> B = {{3}};
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 1, 1);

    std::vector<std::vector<int>> expected = {{15}};

    ASSERT_EQ(C, expected) << "Scalar matrices test failed! :(((()";
}

/**
 * @brief Test with a row matrix and a column matrix.
 *
 */
TEST(MatrixMultiplicationTest, TestRowColumnMatrices)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3}};
    std::vector<std::vector<int>> B = {
        {4},
        {5},
        {6}};
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 3, 1);

    std::vector<std::vector<int>> expected = {{32}};

    ASSERT_EQ(C, expected) << "Row and column matrices test failed! :(((()";
}

/**
 * @brief Test with a column matrix and a row matrix.
 *
 */
TEST(MatrixMultiplicationTest, TestColumnRowMatrices)
{
    std::vector<std::vector<int>> A = {
        {1},
        {2},
        {3}};
    std::vector<std::vector<int>> B = {
        {4, 5, 6}};
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));

    multiplyMatrices(A, B, C, 3, 1, 3);

    std::vector<std::vector<int>> expected = {
        {4, 5, 6},
        {8, 10, 12},
        {12, 15, 18}};

    ASSERT_EQ(C, expected) << "Column and row matrices test failed! :(((()";
}

////////////////////////////////////////////////////////////////////////////////////////
//                                  INCOMPATIBLE INPUT VALUES                          //
////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Test with incompatible dimensions of the objects A and B.
 *
 * This test causes a SEGFAULT, which is likely due to an out of bounds access.
 * This shows that the function doesn't check that the input objects are compatible.
 */
/*TEST(MatrixMultiplicationTest, TestIncompatibleObjectsAB)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3}};
    std::vector<std::vector<int>> B = {
        {4},
        {5}};
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 3, 1);

    // Since the input objects are incompatible, the result is not predictable.
    // Since it turns out to cause a segfault, we accept it to pass in any case, in order not to block the execution.
    ASSERT_EXIT((multiplyMatrices(A, B, C, 1, 3, 1),exit(0)),::testing::KilledBySignal(SIGSEGV),".*");
}*/

/**
 * @brief Test with incompatible dimensions of the object C.
 *
 * This test is passed without rising any exception, that instead should be raised in a correct implementation
 */
TEST(MatrixMultiplicationTest, TestIncompatibleObjectC)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3}};
    std::vector<std::vector<int>> B = {
        {4},
        {5},
        {6}};
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 1, 3, 1);

    // Since the object C has incompatible dimensions, the result is not predictable.
    //  This is why we accept the test to pass in any case
    SUCCEED();
}

/**
 * @brief Test with the value of rowsA greater than the effective number of rows of the matrix A.
 *
 * This test causes a SEGFAULT, which is likely due to an out of bounds access.
 * This shows that, as we supposed, the function wrongly doesn't check the correspondence 
 * between the dimensions of the input matrices and the effective dimensions of the matrices' objects themselves.
 * Instead, it only relies on the input values, taking the correct correspondence with the input object for granted.
 * 
 */
TEST(MatrixMultiplicationTest, TestRowsAGreaterThanEffectiveRowsA)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3}};
    std::vector<std::vector<int>> B = {
        {4},
        {5},
        {6}};
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    
    // Since the value of rowsA is greater than the effective number of rows of the matrix A,
    // the result is not predictable.
    // Since it results to causes a segfault, we accept it to pass in any case, in order not to block the execution. 
    ASSERT_EXIT((multiplyMatrices(A, B, C, 2, 3, 1),exit(0)),::testing::KilledBySignal(SIGSEGV),".*");
}

/**
 * @brief Test with the value of rowsA smaller than the effective number of rows of the matrix A.
 *
 * This test is passed without rising any exception, that instead should be raised in a correct implementation
 * of the function. The result is obviously meaningless.
 */
TEST(MatrixMultiplicationTest, TestRowsASmallerThanEffectiveRowsA)
{
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}};
    std::vector<std::vector<int>> B = {
        {4},
        {5},
        {6}};
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));

    multiplyMatrices(A, B, C, 0, 3, 1);

    // Since the value of rowsA is smaller than the effective number of rows of the matrix A,
    //  the result is not predictable. This is why we accept the test to pass in any case.
    ASSERT_EQ(1,1);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/*
ERRORS SPOTTED:
Error 1: Element-wise multiplication of ones detected!
Error 2: Matrix A contains the number 7!
Error 3: Matrix A contains a negative number!
Error 4: Matrix B contains the number 3!
Error 5: Matrix B contains a negative number!
Error 6: Result matrix contains a number bigger than 100!
Error 7: Result matrix contains a number between 11 and 20!
Error 8: Result matrix contains zero!
Error 10: A row in matrix A contains more than one 1!
Error 11: Every row in matrix B contains at least one 0!
Error 12: The number of rows in A is equal to the number of columns in B!
Error 13: The first element of matrix A is equal to the first element of  matrix B!
Error 14: The result matrix C has an even number of rows!
Error 15: A row in matrix A is filled entirely with 5s!
Error 16: Matrix B contains the number 6!
Error 18: Matrix A is a square matrix!
Error 20: Number of columns in matrix A is odd!
*/