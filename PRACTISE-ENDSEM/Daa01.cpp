// // NON RECURSIVE

#include <iostream>
using namespace std;


// void fibo( int num )
// {

//   int n1 = 0 ;
//   int n2 = 1 ;
//   int sum = 0;
//   for( int i = 0 ; i < num ; i++ )
//     {
//       cout << n1 << "\t ";
//       sum = n1 + n2;
//       n1 = n2;
//       n2 = sum;
//     }

// }


// int main()
// {

//   int  num = 0 ;
  
//   cout<< " Enter the number : ";
//   cin >> num;

//   fibo(num);


//   return 0;
// }



int fibo( int n )
{
  if  ( n == 0 )
    {
      return 0;
    }
  if ( n == 1 )
    {
      return 1;
    }
  return fibo(n-1) + fibo(n-2);
}



int main()
{
  int n ;

  cout << " enter the num ; ";
  cin >> n;

  if (  n < 0 )
    {
      return 1;
    }

  for ( int i = 0 ; i < n ; i++ )
    {
      cout << fibo(i) << "\t";
    }

  cout << endl;

  return 0;
}
