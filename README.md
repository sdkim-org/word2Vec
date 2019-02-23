### 모듈 설치
python 3.6 버전 필요.<br>
numpy, tensorflow 설치 (urllib 는 기본 모듈) <br>
시각화를 위한 모듈 : sklearn, matplotlib, scipy 설치 <br>

<br>

### Step 1. 데이터 다운로드

```python
filename = maybe_download('text8.zip', 31344016)
```
> 파일이 없으면 다운로드 후 파일 이름을 리턴. (두번째 인자는 파일 사이즈)

<br>

```python
vocabulary = read_data(filename)
```
> text 파일을 공백기준(space, enter)으로 split. 결과는 배열에 담긴다.

<br>

### Step 2. 사전을 구축하고 거의 등장하지 않는 단어를 UNK 토큰으로 대체.
UNK는 unknown 약자로 출현 빈도가 낮은 단어들을 모두 대체한다. UNK 갯수는 418391.<br>

```python
data, count, ordered_words = build_dataset(vocabulary, vocabulary_size)
```
> vocabulary 와 bocabulary_size 를 매개변수로 전달한다.
> data : 단어에 대한 인덱스만으로 구성된 리스트<br>
> count : 단어와 빈도 쌍으로 구성된 리스트. 중요한 변수이지만, 이번 코드에서는 사용 안함.<br>
> ordered_words : 빈도에 따라 정렬된 단어 리스트<br>

<br>

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

<br>

### Step 3. skip-gram 모델에 사용할 학습 데이터를 생성할 함수 작성

전체 데이터로부터 minibatch 에 사용할 샘플 데이터를 생성하는 단계로, generate_batch() 를 만드는 것이 목표.<br>
`Step 5` 에 있는 반복문에서 한 번만 사용이 된다.  

```python
batch, labels, data_index = generate_batch(data, batch_size=8, num_skips=2, skip_window=1, data_index=0)
for i in range(8):
    print('{} {} -> {} {}'.format(batch[i],     ordered_words[batch[i]],
                                  labels[i, 0], ordered_words[labels[i, 0]]))
```
> data : 단어 인덱스 리스트
> batch_size : SGD 알고리즘에 적용할 데이터 갯수. 한 번에 처리할 크기.<br>
> num_skips : context window에서 구축할 (target, context) 쌍의 갯수.<br>
> skip_window : skip-gram 모델에 사용할 윈도우 크기.<br>
>               1이라면 목표 단어(target) 양쪽에 1개 단어이므로 context window 크기는 3이 된다. (단어, target, 단어)<br>
>               2라면 5가 된다. (단어 단어 target 단어 단어)<br>
> data_index : 첫 번째 context window에 들어갈 data에서의 시작 위치.<br>

<br>

`컨텍스트(context)` 란 CBOW와 Skip-Gram 모델에서 사용하는 용어로, "계산이 이루어지는 단어들"을 말한다. <br>
컨텍스트는 구둣점으로 구분되어지는 문장(sentence)을 의미하는 것이 아니라 특정 단어 주변에 오는 단어들의 집합을 의미한다.<br> 
"the cat sits on the" 컨텍스트가 있다면, sits라는 단어 양쪽으로 2개의 단어들이 더 있는데, 이들 단어를 모두 합친 5개의 단어가 컨텍스트가 되는 것이다.<br>
"the cat sits on the" 뒤에는 'mat'라는 추가 단어가 올 수 있지만, 컨텍스트에는 포함되지 않는다. <br>
이 경우의 컨텍스트는 목표 단어 양쪽에 2개의 단어만을 허용한 경우이다. <br>

<br>

CBOW 모델은 주변 단어, 다른 말로 맥락(context)으로 타겟 단어(target word)를 예측하는 문제를 푼다. <br>
주변 단어란 보통 타겟 단어의 직전 몇 단어와 직후 몇 단어를 뜻한다. <br>
타겟 단어의 앞 뒤에 있는 단어들을 타겟 단어의 친구들이라고 보는 것이다. 이 주변 단어의 범위를 window라고 부른다.

Reference
[https://dreamgonfly.github.io/machine/learning,/natural/language/processing/2017/08/16/word2vec_explained.html](https://dreamgonfly.github.io/machine/learning,/natural/language/processing/2017/08/16/word2vec_explained.html)


```python
assert batch_size % num_skips == 0
assert num_skips <= 2 * skip_window
```
> condition 이 false 면 AssertionError 를 발생시킨다. 즉 저 조건에 맞지 않으면 종료가 된다. 

<br>

```python
temp = 'batch_size {}, num_skips {}, skip_window {}, data_index {}'
# print(temp.format(batch_size, num_skips, skip_window, data_index))
# 최초 : batch_size 128, num_skips 2, skip_window 1, data_index 0
# 학습 : batch_size 128, num_skips 2, skip_window 1, data_index 640000
#       data_index는 64로 시작해서 64씩 증가한다. 나머지는 변경되지 않는다. (by Step 5)
```

<br>


```python
import collections
skip_window = 1
span = 2 * skip_window + 1  
buffer = collections.deque(maxlen=span)
```
> 현재 `deque` 의 사이즈는 3 <br>

```
>>> buffer
deque([], maxlen=3)
>>> buffer.append(1)
>>> buffer.append(2)
>>> buffer
deque([1, 2], maxlen=3)
>>> buffer.append(3)
>>> buffer
deque([1, 2, 3], maxlen=3)
>>> buffer.append(4)
>>> buffer
deque([2, 3, 4], maxlen=3)
```
> 위와 같은 결과를 얻을 수 있다.<br>
> 즉, 값을 추가하면, 가장 먼저 추가되었던 값이 삭제가 된다. <br>

<br>

```python
for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)   # 다음 단어 인덱스로 이동. len(data) = 17005207
```

> 모듈러를 사용한 이유는 원형큐의 개념을 생각하면 될 것 같다. 




```python
# [출력 결과]
# 3081 originated -> 12 as
# 3081 originated -> 5234 anarchism
# 12 as -> 6 a
# 12 as -> 3081 originated
# 6 a -> 195 term
# 6 a -> 12 as
# 195 term -> 2 of
# 195 term -> 6 a
```


<br><br>

소스코드 출처 : [https://pythonkim.tistory.com/93?category=613486](https://pythonkim.tistory.com/93?category=613486)

