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
        if(nums1.size() > nums2.size())
            return findMedianSortedArrays(nums2, nums1);

        vector<int> v = {(int)nums1.size(), (int)nums2.size(), 0, (int) nums1.size(), 0, 0, 0, 0, 0, 0};
        return medianHelper(nums1, nums2, v);
    }

    double medianHelper(vector<int>& nums1, vector<int>& nums2, vector<int>& v) {
        if(v[0] > v[1]) return -1;
        v[4] = (v[2] + v[3]) / 2;
        v[5] = (v[0] + v[1] + 1) / 2 - v[4];

        v[6] = v[4] == 0 ? INT_MIN : nums1[v[4] - 1];
        v[7] = v[4] == v[0] ? INT_MAX : nums1[v[4]];
        v[8] = v[5] == 0 ? INT_MIN : nums2[v[5] - 1];
        v[9] = v[5] == v[1] ? INT_MAX : nums2[v[5]];

        if(v[6] <= v[9] && v[7] >= v[8]) {
            if((v[0] + v[1]) & 0x1) return max(v[6], v[8]);
            else return (max(v[6], v[8]) + (min(v[7], v[9]))) / 2.0;
        }
        else if(v[6] > v[9]) v[3] = v[4] - 1;
        else v[2] = v[4] + 1;
        return medianHelper(nums1, nums2, v);
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
