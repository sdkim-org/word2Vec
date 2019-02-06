### 모듈 설치
python 3.6 버전 필요.<br>
numpy, tensorflow 설치 (urllib 는 기본 모듈) <br>
시각화를 위한 모듈 : sklearn, matplotlib, scipy 설치 <br>

### Step 1. 데이터 다운로드

```python
filename = maybe_download('text8.zip', 31344016)
```
> 파일이 없으면 다운로드 후 파일 이름을 리턴. (두번째 인자는 파일 사이즈)

```python
vocabulary = read_data(filename)
```
> text 파일을 공백기준(space, enter)으로 split. 결과는 배열에 담긴다.


### Step 2. 사전을 구축하고 거의 등장하지 않는 단어를 UNK 토큰으로 대체.
UNK는 unknown 약자로 출현 빈도가 낮은 단어들을 모두 대체한다. UNK 갯수는 418391.<br>

```python
data, count, ordered_words = build_dataset(vocabulary, vocabulary_size)
```
> vocabulary 와 bocabulary_size 를 매개변수로 전달한다.
> data : 단어에 대한 인덱스만으로 구성된 리스트<br>
> count : 단어와 빈도 쌍으로 구성된 리스트. 중요한 변수이지만, 이번 코드에서는 사용 안함.<br>
> ordered_words : 빈도에 따라 정렬된 단어 리스트<br>

`collections.Counter()` 는 컨테이너에 동일한 값의 자료가 몇개인지를 파악하는데 사용하는 객체로 결과값(return)은 딕셔너리 형태.
```python
# collections.Counter 예제 (1)
# list를 입력값으로 함
import collections
lst = ['aa', 'cc', 'dd', 'aa', 'bb', 'ee']
print(collections.Counter(lst))

# 결과
# Counter({'aa': 2, 'cc': 1, 'dd': 1, 'bb': 1, 'ee': 1})
```
> [참고 https://excelsior-cjh.tistory.com/94](https://excelsior-cjh.tistory.com/94)

<br>

`most_common(n)` 은 가장 많이 출현한 키를 n개 까지 찾아준다. 이때, 공동 x위가 있는 경우에도 n개 만큼 잘리니 주의해야 한다. 
리턴값은 (키, 카운트)의 튜플의 리스트이며, n을 생략하면 전체 키에 대해서 빈도순으로 정렬하여 출력한다.
```python
import collections
 
myList = ['a', 'b', 'c', 'c', 'a', 'a']
myCounter = collections.Counter(myList)
print('myCounter:', myCounter)
# myCounter: Counter({'a': 3, 'c': 2, 'b': 1})
 
print("myCounter['a']:", myCounter['a'])
# myCounter['a']: 3
 
yourList = ['a', 'd', 'c', 'a', 'b']
yourCounter = collections.Counter(yourList)
print('yourCounter:', yourCounter)
# yourCounter: Counter({'a': 2, 'd': 1, 'b': 1, 'c': 1})
 
ourCounter = myCounter + yourCounter
print('ourCounter:', ourCounter)
# ourCounter: Counter({'a': 5, 'c': 3, 'b': 2, 'd': 1})
 
print('ourCounter.most_common(3):', ourCounter.most_common(3))
# ourCounter.most_common(3): [('a', 5), ('c', 3), ('b', 2)]
```
> [참고 https://godoftyping.wordpress.com/2015/04/20/python-카운터/](https://godoftyping.wordpress.com/2015/04/20/python-카운터/)<br>
> [참고 https://soooprmx.com/archives/8602](https://soooprmx.com/archives/8602)

<br>

`List.extend(x)` 는 매개변수로 리스트만 올 수 있으며, 기존 리스트인 `List` 에 `x` 를 추가하게 된다.
```python
list = [('a', 5), ('c', 3), ('b', 2)];
count = [['UNK', -1]];
count.extend(list);
print(count);
# [['UNK', -1], ('a', 5), ('c', 3), ('b', 2)]
``` 
> [참고 https://wikidocs.net/14#extend](https://wikidocs.net/14#extend)<br>

`dictionary = {}` 에는 (word, index) 쌍으로 인덱싱이 되는데, word 는 most_common 에서 추출된 단어들<br>

<br>

`data = []` 에는 `vocabulary` 에 있는 단어 하나하나에 대해 인덱스를 담는다. 
만약 most_common 에 해당하지 않아 인덱스를 부여받지 못한 단어는 0(UNK)으로 기록이 되고, 그 갯수가 카운팅된다. <br>

<br>

결국, `build_dataset()` 는 `data, count, list(dictionary.keys())` 를 리턴한다. 

### Step 3. skip-gram 모델에 사용할 학습 데이터를 생성할 함수 작성




소스코드 출처 : [https://pythonkim.tistory.com/93?category=613486](https://pythonkim.tistory.com/93?category=613486)

