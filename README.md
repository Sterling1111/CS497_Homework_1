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
of array b. Since the algorithm performs a binary search on the smaller array it it
O(log(min{m, n})) = O(log(m + n)). Space complexity is O(1) since constant space is used. 

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

