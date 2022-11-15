#include <iostream>
#include <vector>
#include <queue>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <set>
#include <math.h>

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
    /**
     * Homework 9
     */
    int longestValidParentheses(string s) {
        if(s.size() < 2) return 0;

        vector<int> dp(s.size(), 0);

        dp[1] = (s[0] == '(' and s[1] == ')') ? 2 : 0;
        int max{dp[1]};

        for(int i = 2; i < s.size(); i++) {
            if(s[i] == '(') continue;
            if(s[i-1] == '(') dp[i] = dp[i-2] + 2;
            else if((i - dp[i-1]-1 > -1) and s[i-dp[i-1]-1] == '(') {
                dp[i] = dp[i-1] + 2;
                dp[i] += (i - dp[i-1]-2 > -1) ? dp[i-dp[i-1]-2] : 0;
            }
            max = std::max(max, dp[i]);
        } return max;
    }

    int maxSubArray(vector<int>& nums) {
        int max = nums[0];
        int curr = max;

        for(int i = 1; i < nums.size(); i++) {
            curr = std::max(nums[i], curr + nums[i]);
            max = std::max(max, curr);
        }
        return max;
    }

    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 1));

        for(int i = 1; i < m; i++) {
            for(int j = 1; j < n; j++) {
                dp[i][j] = dp[i][j-1] + dp[i-1][j];
            }
        } return dp[m-1][n-1];
    }

    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        if(obstacleGrid[0][0]) return 0;
        obstacleGrid[0][0] = 1;

        int rows = obstacleGrid.size();
        int cols = obstacleGrid[0].size();


        for(int row = 1; row < rows; row++)
            obstacleGrid[row][0] = (!obstacleGrid[row][0] and obstacleGrid[row-1][0]) ? 1 : 0;


        for(int col = 1; col < cols; col++)
            obstacleGrid[0][col] = (!obstacleGrid[0][col] and obstacleGrid[0][col-1]) ? 1 : 0;

        for(int row = 1; row < rows; row++) {
            for(int col = 1; col < cols; col++) {
                if(obstacleGrid[row][col])
                    obstacleGrid[row][col] = 0;
                else
                    obstacleGrid[row][col] = obstacleGrid[row-1][col] + obstacleGrid[row][col-1];
            }
        } return obstacleGrid[rows-1][cols-1];
    }

    /**
     * Homework 8
     */
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> result(0);
        unordered_map<int, vector<int>> m;
        unordered_map<int, bool> seen;
        vector<int> indegree(numCourses);
        queue<int> q;

        for(const auto& courses : prerequisites) {
            int a = courses[0];
            int b = courses[1];
            m[b].push_back(a);
            indegree[a]++;
        }

        for(int i = 0; i < numCourses; i++) {
            if(!indegree[i]) {
                q.push(i);
                seen[i] = true;
            }
        }

        while(!q.empty()) {
            int node = q.front();
            result.push_back(node);
            q.pop();
            if(m.find(node) != m.end()) {
                for(auto elem : m[node]) {
                    indegree[elem]--;
                    if(!indegree[elem] && !seen[elem]) {
                        q.push(elem);
                        seen[elem] = true;
                    }
                }
            }
        } if(result.size() == numCourses) return result;
        return {};
    }

    int divide(int dividend, int divisor) {
        int result{};
        if(divisor == numeric_limits<int>::min()) return dividend == divisor;
        if(divisor == 1) return dividend;
        if(dividend == numeric_limits<int>::min()) {
            if(divisor == 1) return numeric_limits<int>::min();
            else if(divisor == -1) return numeric_limits<int>::max();
            else {
                dividend += abs(divisor);
                ++result;
            }
        }
        result += exp(log(abs(dividend)) - log(abs(divisor))) + .0000001;
        return dividend > 0 == divisor > 0 ? result : -result;
    }

    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount + 1, amount + 1);
        dp[0] = 0;

        for(int i = 1; i < amount + 1; i++) {
            for(int c : coins) {
                if(i - c >= 0)
                    dp[i] = min(dp[i], 1 + dp[i - c]);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }

    /**
     * Homework 7
     */
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        unordered_map<int, vector<int>> m;
        unordered_map<int, bool> seen;
        vector<int> indegree(numCourses);
        queue<int> q;
        int count = 0;

        for(const auto& courses : prerequisites) {
            int a = courses[0];
            int b = courses[1];
            m[b].push_back(a);
            indegree[a]++;
        }

        for(int i = 0; i < numCourses; i++) {
            if(!indegree[i]) {
                q.push(i);
                seen[i] = true;
            }
        }

        while(!q.empty()) {
            int node = q.front();
            q.pop();
            count++;
            if(m.find(node) != m.end()) {
                for(auto elem : m[node]) {
                    indegree[elem]--;
                    if(!indegree[elem] && !seen[elem]) {
                        q.push(elem);
                        seen[elem] = true;
                    }
                }
            }
        } return count == numCourses;
    }

    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<vector<int>>> graph(n+1);
        vector<int> distance(n+1, numeric_limits<int>::max());
        vector<bool> visited(n+1, false);
        priority_queue<vector<int>, vector<vector<int>>, greater<>> pq;
        for(const auto& time : times)
            graph.at(time.at(0)).push_back({time.at(1), time.at(2)});

        distance[k] = 0;
        pq.push({0, k});
        int u, v, w;

        while(!pq.empty()) {
            u = pq.top().at(1);
            pq.pop();
            if(visited[u]) continue;
            visited[u] = true;
            for(auto& elem : graph[u]) {
                v = elem.at(0);
                w = elem.at(1);
                if(distance.at(v) > distance.at(u) + w) {
                    distance[v] = distance[u] + w;
                    pq.push({distance[v], v});
                }
            }
        }
        int result = *max_element(distance.begin() + 1, distance.end());
        if(result == numeric_limits<int>::max()) return -1;
        return result;
    }

    int minCostToSupplyWater(int n, vector<int>& wells, vector<vector<int>>& pipes) {
        vector<vector<int>> edges (pipes.size());

        for(int i = 0; i < pipes.size(); i++) edges[i] = pipes[i];
        for(int i = 0; i < wells.size(); i++) edges.push_back({0, i+1, wells[i]});

        sort(edges.begin(), edges.end(), [](const auto& v1, const auto& v2) {
            return v1[2] < v2[2];
        });

        parent.clear();
        parent.resize(edges.size());

        for(int i = 0; i < parent.size(); i++) parent[i] = i;
        int result{0}, count{0};

        for(const auto& elem : edges) {
            auto parent1{findParent(elem[0])}, parent2{findParent(elem[1])};
            if(parent1 != parent2) {
                result += elem[2];
                parent[parent2] = parent1;
                count++;
                if(count == n) return result;
            }
        }
        return -1;
    }

    /**
     *
     */
    bool dfs(vector<pair<bool, vector<int>>>& graph, int currNode, int destination) {
        if(currNode == destination) return true;

        auto & [seen, adjacentNodes] = graph[currNode];

        if(!seen) {
            seen = true;
            for(auto node : adjacentNodes) {
                if (dfs(graph, node, destination)) return true;
            }
        }
        return false;
    }

    bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
        vector<pair<bool, vector<int>>> graph(n, {false, {}});

        for(auto& edge : edges) {
            int a = edge[0], b = edge[1];
            graph[a].second.push_back(b);
            graph[b].second.push_back(a);
        }

        return dfs(graph, source, destination);
    }

    int longestCycle(vector<int>& edges) {
        int result{-1};
        //{distance, source node}
        vector<pair<int, int>> v(edges.size(), {-1, -1});

        for (int i = 0; i < edges.size(); ++i) {
            int dist{0}, j{i};

            //while we have not gotten to the end of the graph
            while(j != -1) {
                auto & [sourceDist, sourceNode] = v[j];

                if(sourceDist == -1) v[j] = {dist++, i};
                else {
                    if(sourceNode == i) result = std::max(result, dist - sourceDist);
                    break;
                }
                j = edges[j];
            }
        }
        return result;
    }

    vector<int> parent;

    int findParent(int i) {
        while(i != parent[i]) i = parent[i];
        return i;
    }

    int minimumCost(int n, vector<vector<int>>& connections) {
        std::sort(connections.begin(), connections.end(), [] (const auto& v1, const auto& v2) {
            return v1[2] < v2[2];
        });
        parent.resize(n+1);
        for (int i = 1; i <= n; ++i) parent[i] = i;
        int result{0}, count{1};
        for(const auto& elem : connections) {
            auto parent1{findParent(elem[0])}, parent2{findParent(elem[1])};
            if(parent1 != parent2) {
                result += elem[2];
                parent[parent2] = parent1;
                count++;
                if(count == n) return result;
            }
        } return -1;
    }

    unordered_set<string> validExpressions;
    unsigned maxSizeAccepted{};

    void rec(const string& s, int index, int leftCount, int rightCount, string& expr) {
        if(index == s.size()) { //we have decided what to do with each elem. Keep or remove
            if(leftCount == rightCount) { //parentheses are balanced
                if(expr.size() >= maxSizeAccepted) {
                    if(expr.size() > maxSizeAccepted) {
                        validExpressions.clear();
                        maxSizeAccepted = expr.size();
                    }
                    validExpressions.insert(expr);
                }
            }
        } else { //we have not processed the entire string yet
            auto curr = s.at(index);

            if(curr != '(' and curr != ')') {
                expr.push_back(curr);
                rec(s, index+1, leftCount, rightCount, expr);
                expr.pop_back();
            } else {
                //in this one we dont add no matter what it is.
                rec(s, index+1, leftCount, rightCount, expr);
                expr.push_back(curr);
                if(curr == '(') {
                    rec(s, index+1, leftCount+1, rightCount, expr);
                }
                else if(leftCount > rightCount) {
                    rec(s, index+1, leftCount, rightCount+1, expr);
                }
                expr.pop_back();
            }
        }
    }

    vector<string> removeInvalidParentheses(string s) {
        vector<string> results;
        string expr;
        rec(s, 0, 0, 0, expr);
        for (const auto& elem : validExpressions) {
            results.push_back(elem);
        }
        return results;
    }

    vector<int> vec1;

    void inorder(TreeNode* root) {
        if(!root) return;
        inorder(root->left);
        vec1.push_back(root->val);
        inorder(root->right);
    }

    int getMinimumDifference(TreeNode* root) {
        inorder(root);
        int minDif = numeric_limits<int>::max();
        for(int i = 0; i < vec1.size()-1; i++) {
            minDif = min(minDif, vec1[i+1] - vec1[i]);
        }
        return minDif;
    }

    typedef int nodeType, maskType, distType;
    typedef pair<nodeType, maskType> stateType;

    int shortestPathLength(vector<vector<int>>& graph) {
        auto size = graph.size();
        maskType endMask = (1 << size) - 1;
        queue<pair<distType, stateType>> q;
        set<stateType> s;

        for (int i = 0; i < size; ++i) {
            q.push({0, {i, 1 << i}});
            s.insert({i, 1 << i});
        }
        while(!q.empty()) {
            auto curr{q.front()};
            q.pop();
            nodeType node{curr.second.first};
            distType dist{curr.first};
            maskType mask{curr.second.second};

            for(nodeType n : graph[node]) {
                maskType updatedMask = mask | (1 << n);
                if(updatedMask == endMask) return dist+1;
                else if(s.find({n, updatedMask}) != end(s)) continue;
                else {
                    q.push({dist+1, {n, updatedMask}});
                    s.insert({n, updatedMask});
                }
            }
        }
        return 0;
    }

    bool dfs(vector<int>& result, int n, int curr) {
        if(curr > n) return false;

        result.push_back(curr);

        for(int i = 0; i < 10; i++) {
            if(!dfs(result, n, curr * 10 + i)) return result.size() < n;
        } return result.size() < n;
    }

    vector<int> lexicalOrder(int n) {
        vector<int> result;
        int i {1};
        while(dfs(result, n, i++));
        return result;
    }

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

template <typename T>
void print(vector<T> v) {
    for (T elem : v) {
        std::cout << elem << " " << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << round((exp(log(231) - log(3)))) << std::endl;
    int num = log(5);
    Solution sol;
    vector<int> v1 = {};
    vector<int> v2 = {2, 3};
    vector<vector<int>> v3 = {{}};
    vector<int> v4 = {3, 3, 4, 2, 3};
    vector<vector<int>> v5 = {{1, 2, 5}, {1, 3, 6}, {2, 3, 1}};
    vector<vector<int>> v6 = {{0, 1}, {1, 2}, {2, 0}};
    print(sol.topKFrequent(v1, 0));
    print(sol.findClosestElements(v1, 0, 0));
    print(sol.peekTopK(v1, 0));
    std::cout << sol.shortestSubarray(v1, 0) << std::endl;
    print(sol.kthSmallestPrimeFraction(v2, 1));
    print(sol.removeInvalidParentheses("()"));
    TreeNode node(5);
    TreeNode node2(6);
    node.left = &node2;
    std::cout << sol.getMinimumDifference(&node) << std::endl;
    std::cout << sol.shortestPathLength(v3) << std::endl;
    std::cout << sol.maxPathSum(&node) << std::endl;
    print(sol.lexicalOrder(3));
    std::cout << sol.longestCycle(v4) << std::endl;
    std::cout << sol.minimumCost(3, v5) << std::endl;
    std::cout << sol.validPath(3, v6, 0, 2) << std::endl;
    vector<int> wells = {1, 2, 2};
    vector<vector<int>> pipes = {{1, 2, 1}, {2, 3, 1}};
    std::cout << sol.minCostToSupplyWater(3, wells, pipes) << std::endl;
    vector<vector<int>> prereq = {{1, 0}};
    vector<vector<int>> vae = {{2, 1, 1}, {2, 3, 1}, {3, 4, 1}};
    std::cout << sol.networkDelayTime(vae, 4, 2) << std::endl;
    std::cout << sol.canFinish(2, prereq) << std::endl;
    return 0;
}
