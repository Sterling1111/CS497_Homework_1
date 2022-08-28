#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> map;
        int complement;

        for(int i = 0; i < nums.size(); i++) {
            complement = target - nums[i];
            if(map.find(complement) != map.end())
                return {map[complement], i};
            map.insert({nums[i], i});
        }
        return {-1, -1};
    }

    vector<int> searchRange(vector<int>& nums, int target) {
        int start = 0, end = nums.size() - 1;
        start = minIndex(nums, target, start, end);
        if(start == -1)
            return {-1, -1};
        return {start, maxIndex(nums, target, start, end)};
    }

    int minIndex(vector<int>& nums, int target, int start, int end) {
        int mid{};

        while(start <= end) {
            mid = (start + end) / 2;
            if(nums[mid] < target)
                start = mid + 1;
            else if(nums[mid] > target)
                end = mid - 1;
            else if(mid == start || nums[end = mid-1] != target)
                return mid;
        }
        return -1;
    }

    int maxIndex(vector<int>& nums, int target, int start, int end) {
        int mid{};

        while(start <= end) {
            mid = (start + end) / 2;
            if(nums[mid] > target)
                end = mid -1;
            else if(nums[mid] < target)
                start = mid + 1;
            else if(mid == end || nums[start = mid+1] != target)
                return mid;
        }
        return -1;
    }

    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        if(nums1.size() > nums2.size()) {
            return findMedianSortedArrays(nums2, nums1);
        }

        int m = nums1.size();
        int n = nums2.size();

        int start = 0;
        int end = m;

        while(start <= end) {
            int part1 = (start + end) / 2;
            int part2 = (m + n + 1) / 2 - part1;

            int maxLeftPart1 = part1 == 0 ? INT_MIN : nums1[part1 - 1];
            int minRightPart1 = part1 == m ? INT_MAX : nums1[part1];
            int maxLeftPart2 = part2 == 0 ? INT_MIN : nums2[part2 - 1];
            int minRightPart2 = part2 == n ? INT_MAX : nums2[part2];

            if(maxLeftPart1 <= minRightPart2 && minRightPart1 >= maxLeftPart2) {
                //if combined elements are odd parity
                if((m + n) & 0x1) {
                    return max(maxLeftPart1, maxLeftPart2);
                } else {
                    return (max(maxLeftPart1, maxLeftPart2) + (min(minRightPart1, minRightPart2))) / 2.0;
                }
            }
            else if(maxLeftPart1 > minRightPart2) {
                end = part1 - 1;
            } else {
                start = part1 + 1;
            }
        }
        return -1;
    }

    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* temp{head};
        int size{1};

        while((temp = temp->next)) size++;

        if(size == 1) return nullptr;
        if(size == n) return head->next;

        temp = head;

        for(int i = 0; i < size-n-1; i++) temp = temp->next;

        if(n == 1) temp->next = nullptr;
        else temp->next = temp->next->next;

        return head;
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        vector<ListNode*> vec;
        for(auto elem : lists) {
            while(elem) {
                vec.push_back(elem);
                elem = elem->next;
            }
        }
        sort(vec.begin(), vec.end(), [](ListNode* ln1, ListNode* ln2) {
            return ln1->val < ln2->val;
        });
        int size = vec.size() - 1;
        for(int i = 0; i < size; i++)
            vec[i]->next = vec[i + 1];
        if(vec.empty()) return nullptr;
        vec[vec.size() - 1]->next = nullptr;
        return vec[0];
    }
};


int main() {
    std::cout << "Hello, world" << std::endl;
    return 0;
}
