// Practical 3 — 0/1 Knapsack using Dynamic Programming
// dp[i][w] = maximum profit using the first i items with capacity w

#include <iostream>
#include <vector>
using namespace std;

int main()
{
    int n;

    cout << "Enter number of items: ";
    cin >> n;

    if (n <= 0)
    {
        cout << "Invalid number of items!";
        return 1;
    }

    vector<int> weight(n), profit(n);

    cout << "\nEnter weight and profit of each item:\n";
    for (int i = 0; i < n; i++)
        cin >> weight[i] >> profit[i];

    int W;

    cout << "Enter knapsack capacity: ";
    cin >> W;

    // dp[i][w] = maximum profit using the first i items with capacity w
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++)
    {
        for (int w = 0; w <= W; w++)
        {
            // Option 1: skip item i
            dp[i][w] = dp[i - 1][w];

            // Option 2: take item i (if it fits)
            if (weight[i - 1] <= w)
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weight[i - 1]] + profit[i - 1]);
        }
    }

    cout << "\nMaximum profit: " << dp[n][W] << endl;

    // Trace back the table to find which items were taken
    vector<int> taken;
    int w = W;

    for (int i = n; i >= 1; i--)
    {
        if (dp[i][w] != dp[i - 1][w])
        {
            taken.push_back(i);   // item i was taken
            w -= weight[i - 1];
        }
    }

    cout << "Selected items (item: weight, profit): ";

    for (int t = taken.size() - 1; t >= 0; t--)
    {
        int i = taken[t];
        cout << i << ": (" << weight[i - 1] << ", " << profit[i - 1] << ") ";
    }

    cout << endl;

    return 0;
}
