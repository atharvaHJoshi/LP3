
#include <bits/stdc++.h>
using namespace std;

int detSteps = 0;
int randSteps = 0;

// Deterministic Quick Sort
int partition(vector<int>& a, int low, int high)
{
    int pivot = a[high];  // Always last element
    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        detSteps++;

        if (a[j] <= pivot)
        {
            i++;
            swap(a[i], a[j]);
        }
    }

    swap(a[i + 1], a[high]);

    return i + 1;
}

void quickSort(vector<int>& a, int low, int high)
{
    if (low < high)
    {
        int p = partition(a, low, high);

        quickSort(a, low, p - 1);
        quickSort(a, p + 1, high);
    }
}

// Randomized Quick Sort
int randomPartition(vector<int>& a, int low, int high)
{
    // Randomly select pivot
    int randomIndex = low + rand() % (high - low + 1);

    // Swap random pivot with last element
    swap(a[randomIndex], a[high]);

    int pivot = a[high];
    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        randSteps++;

        if (a[j] <= pivot)
        {
            i++;
            swap(a[i], a[j]);
        }
    }

    swap(a[i + 1], a[high]);

    return i + 1;
}

void randomizedQuickSort(vector<int>& a, int low, int high)
{
    if (low < high)
    {
        int p = randomPartition(a, low, high);

        randomizedQuickSort(a, low, p - 1);
        randomizedQuickSort(a, p + 1, high);
    }
}

int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);

    cout << "Enter array elements: ";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    vector<int> a1 = arr;
    vector<int> a2 = arr;

    // Deterministic Quick Sort
    quickSort(a1, 0, n - 1);

    // Randomized Quick Sort
    srand(time(0));
    randomizedQuickSort(a2, 0, n - 1);

    cout << "\nSorted Array:\n";
    for (int x : a1)
        cout << x << " ";

    cout << "\n\nDeterministic Quick Sort Steps: "
         << detSteps;

    cout << "\nRandomized Quick Sort Steps: "
         << randSteps;

    cout << endl;

    return 0;
}
