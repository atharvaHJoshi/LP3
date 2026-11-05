
#include <iostream>
#include <vector>
using namespace std;

// Function to display the chessboard
void displayBoard(vector<vector<int>>& board, int n)
{
    cout << "\nFinal N-Queens Matrix:\n\n";

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << board[i][j] << " ";
        }

        cout << endl;
    }
}

// Function to check whether placing a Queen is safe
bool isSafe(vector<vector<int>>& board, int row, int col, int n)
{
    // Check same column in previous rows
    for (int i = 0; i < row; i++)
    {
        if (board[i][col] == 1)
            return false;
    }
    
    // Check upper-left diagonal
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
    {
        if (board[i][j] == 1)
            return false;
    }

    // Check upper-right diagonal
    for (int i = row - 1, j = col + 1;
         i >= 0 && j < n; i--, j++)
    {
        if (board[i][j] == 1)
            return false;
    }

    return true;
}

// Recursive Backtracking Function
bool solveNQueens(vector<vector<int>>& board, int row, int n)
{
    // All Queens have been placed
    if (row == n)
        return true;

    // Try placing Queen in every column
    for (int col = 0; col < n; col++)
    {
        if (isSafe(board, row, col, n))
        {
            // Place Queen
            board[row][col] = 1;

            // Move to next row
            if (solveNQueens(board, row + 1, n))
                return true;

            // Backtrack: Remove Queen
            board[row][col] = 0;
        }
    }

    return false;
}

// Main Function
int main()
{
    int n;

    cout << "Enter the size of chessboard (N): ";
    cin >> n;

    if (n <= 0)
    {
        cout << "Invalid board size!";
        return 1;
    }

    vector<vector<int>> board(n, vector<int>(n, 0));

    int row, col;

    cout << "Enter first Queen position (row and column): ";
    cin >> row >> col;

    // Convert to zero-based indexing
    row--;
    col--;

    if (row < 0 || row >= n || col < 0 || col >= n)
    {
        cout << "Invalid Queen position!";
        return 1;
    }

    // Place the first Queen
    board[row][col] = 1;

    // Solve the remaining Queens
    if (solveNQueens(board, row + 1, n))
    {
        displayBoard(board, n);
    }
    else
    {
        cout << "\nNo solution exists for this first Queen position.";
    }

    return 0;
}
