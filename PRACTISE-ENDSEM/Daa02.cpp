
#include <bits/stdc++.h>
using namespace std;

// Structure for Huffman Tree Node
struct Node
{
    char data;
    int freq;

    Node* left;
    Node* right;

    Node(char ch, int freq)
    {
        data = ch;
        this->freq = freq;

        left = nullptr;
        right = nullptr;
    }
};

// Comparator for Min Heap
struct Compare
{
    bool operator()(Node* a, Node* b)
    {
        return a->freq > b->freq;
    }
};

// Function to generate Huffman Codes
void generateCodes(Node* root, string code)
{
    if (root == nullptr)
        return;

    // Check leaf node
    if (root->left == nullptr && root->right == nullptr)
    {
        cout << root->data << "\t\t";
        cout << (code.empty() ? "0" : code) << endl;

        return;
    }

    // Left = 0
    generateCodes(root->left, code + "0");

    // Right = 1
    generateCodes(root->right, code + "1");
}

// Function to build Huffman Tree
Node* build_Huffman_Tree(vector<char>& characters,
                         vector<int>& freq, int n)
{
    priority_queue<Node*, vector<Node*>, Compare> minHeap;

    // Insert all characters into Min Heap
    for (int i = 0; i < n; i++)
    {
        minHeap.push(new Node(characters[i], freq[i]));
    }

    // Build Huffman Tree using Greedy Strategy
    while (minHeap.size() > 1)
    {
        // Extract two minimum frequency nodes
        Node* left = minHeap.top();
        minHeap.pop();

        Node* right = minHeap.top();
        minHeap.pop();

        // Create parent node
        Node* parent = new Node('\0', left->freq + right->freq);

        parent->left = left;
        parent->right = right;

        // Insert merged node back into Min Heap
        minHeap.push(parent);
    }

    // Return root node
    return minHeap.top();
}

// Function to delete Huffman Tree
void deleteTree(Node* root)
{
    if (root == nullptr)
        return;

    deleteTree(root->left);
    deleteTree(root->right);

    delete root;
}

// Main Function
int main()
{
    int n;

    cout << "Enter the number of characters: ";
    cin >> n;

    if (n <= 0)
    {
        cout << "INVALID NUMBER OF CHARACTERS";
        return 1;
    }

    vector<char> characters(n);
    vector<int> freq(n);

    cout << "\nEnter character and frequency:\n";

    for (int i = 0; i < n; i++)
    {
        cin >> characters[i] >> freq[i];

        if (freq[i] <= 0)
        {
            cout << "Frequency must be positive!";
            return 1;
        }

        // Check duplicate characters
        for (int j = 0; j < i; j++)
        {
            if (characters[i] == characters[j])
            {
                cout << "Duplicate character entered!";
                return 1;
            }
        }
    }

    // Build Huffman Tree
    Node* root = build_Huffman_Tree(characters, freq, n);

    // Display Huffman Codes
    cout << "\nHuffman Codes:\n";
    cout << "Character\tCode\n";

    generateCodes(root, "");

    // Free allocated memory
    deleteTree(root);

    return 0;
}
