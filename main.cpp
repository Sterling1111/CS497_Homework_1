#include <iostream>
#include <vector>
#include <queue>
#include <deque>
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

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr){}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct Bucket{
    int min{numeric_limits<int>::min()};
    int max{numeric_limits<int>::max()};
    bool empty{true};
};

class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        vector<int> kMostFrequent;
        unordered_map<int, int> mp;
        using mpType = pair<int, int>;  //first = value, second = frequency
        auto compare = [](const mpType& p1, const mpType& p2) {return p1.second > p2.second;};
        priority_queue<mpType, vector<mpType>, decltype(compare)> pq(compare);   //min heap

        if(k == nums.size()) return nums;
        for (auto elem : nums) mp[elem]++;

        for(auto elem : mp) {
            if(pq.size() < k) pq.push(elem);
            else if(pq.top().second < elem.second) {
                pq.pop();
                pq.push(elem);
            }
        }
        while(!pq.empty()) {
            kMostFrequent.emplace_back(pq.top().first);
            pq.pop();
        }
        return kMostFrequent;
    }

    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
        vector<int> v(k);
        auto lb = lower_bound(arr.begin(), arr.end(), x);
        if(lb == arr.begin()) copy(arr.begin(), arr.begin() + k, v.begin());
        else if(lb == arr.end()) copy(arr.end() - k, arr.end(), v.begin());
        else {
            auto left = lb, right = lb;
            int size{};
            while(size++ < k and left > arr.begin() and right < arr.end()) {
                if(abs(*(left-1) - x) <= abs(*right - x)) --left;
                else ++right;
            }
            if(left == arr.begin()) copy(arr.begin(), arr.begin() + k, v.begin());
            else if(right == arr.end()) copy(arr.end() - k, arr.end(), v.begin());
            else copy(left, right, v.begin());
        }
        return v;
    }

    vector<int> peekTopK(const vector<int>& A, int k) {
        auto size = A.size();
        if(!size) return {};
        vector<int> result(k);
        typedef pair<int, int> indexValuePair;
        auto compare = [](const indexValuePair& p1, const indexValuePair& p2) {
            return p1.second < p2.second;}; //max heap
        priority_queue<indexValuePair, vector<indexValuePair>, decltype(compare)> pq(compare);

        pq.push({0, A[0]});

        for (int i = 0; i < k; ++i) {
            auto[index, value] = pq.top();
            pq.pop();
            result[i] = value;
            int leftIndex = index + index + 1, rightIndex = index + index + 2;
            if(leftIndex < size) pq.push({leftIndex, A[leftIndex]});
            else continue;
            if(rightIndex < size) pq.push({rightIndex, A[rightIndex]});
        }
        return result;
    }

    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        struct Fraction {
            int n_index, d_index;
            double value;
            Fraction(int num_index, int den_index, double val) :
                    n_index{num_index}, d_index{den_index}, value{val}{}
            bool operator<(const Fraction& f) const {return value > f.value;}
        };
        priority_queue<Fraction> pq;
        auto size = arr.size();

        for (int i = 0; i < size-1; ++i) pq.push(Fraction(i, size-1, double(arr[i]) / arr[size-1]));

        for (int i = 0; i < k-1; ++i) {
            Fraction f = pq.top();
            pq.pop();
            pq.push(Fraction(f.n_index, f.d_index-1, double(arr[f.n_index]) / arr[f.d_index-1]));
        }
        return {arr[pq.top().n_index], arr[pq.top().d_index]};
    }

    vector<int> v1;

    void convertListToVector(ListNode* head) {
        while(head) {
            v1.push_back(head->val);
            head = head->next;
        }
    }

    TreeNode* sortedListToBSTRec(int start, int end) {
        if(start > end) return nullptr;

        int mid = (start + end) / 2;
        TreeNode* node = new TreeNode(v1[mid]);

        if(start == end) return node;

        node->left = sortedListToBSTRec(start, mid - 1);
        node->right = sortedListToBSTRec(mid + 1, end);
        return node;
    }

    TreeNode* sortedListToBST(ListNode* head) {
        convertListToVector(head);
        return sortedListToBSTRec(0, (int)v1.size() - 1);
    }

    unordered_map<int, int> inorderMap;

    TreeNode* buildTreeRec(vector<int>& preorder, int& index, int start, int end) {
        if(start > end) return nullptr;
        int val = preorder[index++];

        TreeNode* root = new TreeNode(val);

        root->left = buildTreeRec(preorder, index, start, inorderMap[val] - 1);
        root->right = buildTreeRec(preorder, index, inorderMap[val] + 1, end);
        return root;
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int preorderIndex{0};
        for(int i = 0; i < inorder.size(); i++) {
            inorderMap[inorder[i]] = i;
        }
        return buildTreeRec(preorder, preorderIndex, 0, (int)inorder.size() - 1);
    }

    int maxSum = numeric_limits<int>::min();

    int maxPathSumRec(TreeNode* root) {
        if(!root) return 0;

        int maxLeft = max(0, maxPathSumRec(root->left));
        int maxRight = max(0, maxPathSumRec(root->right));

        maxSum = max(root->val + maxLeft + maxRight, maxSum);
        return max(root->val + maxLeft, root->val + maxRight);
    }

    int maxPathSum(TreeNode* root) {
        maxPathSumRec(root);
        return maxSum;
    }

    vector<int> largestValues(TreeNode* root) {
        if(!root) return {};
        vector<int> largest;
        queue<TreeNode*> level;
        level.push(root);

        while(!level.empty()) {
            auto size{level.size()};
            int max = numeric_limits<int>::min();
            while(size--) {
                TreeNode* node{level.front()};
                max = std::max(max, node->val);
                if(node->left) level.push(node->left);
                if(node->right) level.push(node->right);
                level.pop();
            }
            largest.push_back(max);
        }
        return largest;
    }

    vector<TreeNode*> v2;

    void fillVectorInorder(TreeNode* root) {
        if(!root) return;
        fillVectorInorder(root->left);
        v2.push_back(root);
        fillVectorInorder(root->right);
    }

    TreeNode* balanceBSTRec(int start, int end) {
        if(start > end) return nullptr;
        int mid = (start + end) / 2;
        v2[mid]->left = balanceBSTRec(start, mid - 1);
        v2[mid]->right = balanceBSTRec(mid + 1, end);
        return v2[mid];
    }

    TreeNode* balanceBST(TreeNode* root) {
        fillVectorInorder(root);
        return balanceBSTRec(0, (int)v2.size() - 1);
    }

    int majorityElement(vector<int>& nums) {
        unordered_map<int, int> mp;
        for(int elem : nums) mp[elem]++;
        return max_element(mp.begin(), mp.end(), [](auto p1, auto p2) {
            return p1.second < p2.second;
        })->first;
    }

    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<>> pq;
        for (int i = 0; i < k; ++i)
            pq.push(nums[i]);
        for (int i = k; i < nums.size(); ++i) {
            if(nums[i] > pq.top()) {
                pq.pop();
                pq.push(nums[i]);
            }
        }
        return pq.top();
    }

    int maximumGap(vector<int>& nums) {
        if(nums.size() < 2) return 0;

        auto[minElem, maxElem] = minmax_element(nums.begin(), nums.end());

        int minimalMaxGap = (*maxElem - *minElem) / ((int)nums.size() - 1);
        int bucketSize = minimalMaxGap + 1;
        int bucketNum = ((*maxElem - *minElem) / bucketSize) + 1;
        vector<Bucket> buckets(bucketNum);

        for (auto num : nums) {
            Bucket& bucket = buckets[(num - *minElem) / bucketSize];
            bucket.empty = false;
            bucket.max = max(bucket.max, num);
            bucket.min = min(bucket.min, num);
        }

        int maxGap = minimalMaxGap, prevMax = *minElem;

        for(const auto& bucket : buckets) {
            if(bucket.empty) continue;
            maxGap = max(maxGap, bucket.min - prevMax);
            prevMax = bucket.max;
        }
        return maxGap;
    }

    string removeDuplicateLetters(string s) {
        if(s.empty()) return s;
        int count[26] = {};
        int index{};
        for(auto elem : s) count[elem - 'a']++;
        for (int i = 0; i < s.size(); ++i) {
            if(s[i] < s[index]) index = i;
            if(!(--count[s[i] - 'a'])) break;
        }
        char letter{s[index]};
        s = s.substr(index + 1);
        s.erase(remove(s.begin(), s.end(), letter), s.end());
        return letter + removeDuplicateLetters(s);
    }

    int shortestSubarray(vector<int>& nums, int k) {
        int size = (int)nums.size();
        int minLength{(int)nums.size() + 1};
        long long sum{};
        typedef tuple<long long, long long, bool> dqNode;
        deque<dqNode> dq;

        for (int i = 0; i < size; ++i) {
            sum += nums[i];
            if(sum >= k)
                minLength = min(minLength, i + 1);

            dqNode is = {0, 0, false};
            while(!dq.empty() && sum - get<1>(dq.front()) >= k) {
                is = dq.front();
                get<2>(is) = true;
                dq.pop_front();
            }
            if(get<2>(is))
                minLength = min((long long)minLength, i - get<0>(is));
            while(!dq.empty() and sum <= get<1>(dq.back()))
                dq.pop_back();
            dq.emplace_back(i, sum, true);
        }
        return minLength == (int)nums.size() + 1 ? -1 : minLength;
    }

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

        v[6] = v[4] == 0 ? numeric_limits<int>::min() : nums1[v[4] - 1];
        v[7] = v[4] == v[0] ? numeric_limits<int>::max() : nums1[v[4]];
        v[8] = v[5] == 0 ? numeric_limits<int>::min() : nums2[v[5] - 1];
        v[9] = v[5] == v[1] ? numeric_limits<int>::max() : nums2[v[5]];

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
    Solution sol;
    vector<int> v = {1, 13, 17, 59};
    sol.kthSmallestPrimeFraction(v, 6);

    return 0;
}
