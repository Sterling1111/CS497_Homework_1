# CS497 Homework 1
## Two Sum
The algorithm iterates through nums and checks for the appropriate complement in a hash map of previously
seen nums. If found the indices are returned otherwise the current element is added to the hashmap.
Iterating through nums is O(n) and inserting and searching a hash map is O(1). So the time complexity
of the algorithm is O(n). We need to store the elements of nums in a hash map so space complexity is O(n).

## Find First and Last Position of Element in Sorted Array
The algorithm performs two binary searches. One to find the minimum index
of the target value and one to find the maximum index of the target value.
The main distinction of these binary searches from a regular binary search
is that when the element is found the first check either the right of left
element to determine if the current element in max or min index. If not the
binary search continues to the left or right. Since binary search divides
the search space roughly in half on each iteration it is O(log n). The space
complexity is O(1) as constant space is used. 

## Median of Two Sorted Arrays
The algorithm partitions both of the arrays so that the sum of the 
number of elements on the left partitions are the same as the sum of the 
number of elements on the right partitions(roughly depending on even or odd).
The goal is to shift the partitions with the goal of maintaining this 
property while also gaining the property that the max elem in left partition
of array a <= the min element in the right part of array b and min element in
the right of array a >= the max element in the left of array b. If these two 
properties are true then we can immediately compute the median. The partitions
are adjusted by removing elements from the smaller array by comparing the max
element in the right party of array a to the minimum element in the right part 
of array b. Since the algorithm performs a binary search on the smaller array it is
O(log(min{m, n})) = O(log(m + n)). Space complexity is O(log(m + n)) as a recursive call is made
for each slicing of the search space in half. 

## Remove Nth Node From End of List
The algorithm computes the number of elements in the list by traversing to the
end. Then the nth node from the end can be deleted by traversing to just before the correct
element and setting its next pointer to next pointer of the element to be deleted. 
The list must be traversed and so time complexity is O(n). Space complexity is
O(1) since constant space is used. 

## Merge k Sorted Lists
The algorithm puts all iterates through all the lists and puts all of their
nodes into an array. Then the array is sorted and reconstructed in a linked list.
Time complexity to insert the list into a vector is O(n). C++ 11 and beyond guarantee 
O(n log n) time complexity for std::sort(). Time complexity to reconstruct the list
is O(n) and so overall time complexity is O(n). Space complexity to store the elements 
of all the lists is O(n). Sorting is not guaranteed space complexity, but it is probably 
not worse than O(n) and so overall space complexity is O(n). 

# CS497 Homework 2

## Majority Element
The algorithm puts all the elements of nums into an unordered map with the key as the
value and the value as the frequency of occurrence. Then return the value that occurs
most frequently. Insertion is O(1) and it occurs n times. Searching all elements for
the max is O(n) and so time complexity is O(n). Space is O(n) for the map of n elements. 

## Kth Largest Element in an Array
The algorithm puts all elements of nums into a priority queue with min element at top and
size of k. Then the top element is returned. It is O(log n) for inserting into a heap and 
insertion occurs O(n) times with a heap of size k. So time complexity is O(n log k). Space
complexity is O(k) for the priority queue. 

## Maximum gap
The algorithm computes the minimum possible max gap and then puts all elements into buckets such
that none of the elements in the same bucket are farther away than the min possible max gap. So if
the answer is larget than that it must occur between the max and min of adjacent non-empty buckets.
There are n -1 buckets so space complexity is O(n). Time complexity is O(n) for placing n elements
into buckets and for iterating through the buckets. 

## Remove Duplicate Letters
The algorithm finds the left-most element that cannot be deleted either because it is the only
of its kind remaining or it is less than all the other elements in the string. Then it removes
all elements to the left of it and all of its kind to the right. Then it calls itself on the right 
remaining part of the string. There are max of 26 function calls, and they do n work and allocate n 
memory each time so the time complexity is O(n) and the space complexity is O(n).

## Shortest Sub-Array with Sum at Least K
The algorithm uses the sliding window technique except we need to make sure that eliminating 
previous elements always results in a decrease. So we use a deque where each element is a region
in the array where there is increase. If we encounter a negative element we remove elements from
the deque from the end until we pay off the debt. Then the sliding window technique will work. The time
is O(n) as we iterate through the elements once and only put 1 element in the deque and if the inner
loops ever run they remove one element. The space complexity is O(n) for the dequeue. 

