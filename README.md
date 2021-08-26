# # Logistic Regression
> Logistic Regression 구현해보기

앞서 배웠던 Linear Regression과 달리 Logistic Regression은 어떤 방법으로 학습하는지 그래프와 그림을 통해 확인합니다.


![](../header.png)

## 소개

  이 Repository에서는 경향성이 있는 임의의 2차원 데이터를 400개 만들고 학습하여 이를 대표하는 선형함수(Hypothesis)를 찾는 과정을 쉽게 볼 수 있도록 만들었습니다. Linear Regression은 scikit learn와 같은 라이브러리에서 쉽게 가져와 사용할 수 있습니다. 하지만 Linear Regression을 직접 구현해보고 동작원리를 제대로 파악하지 않는다면 복잡한 상황에서 제대로 활용하기 어렵습니다.  Linear Regression의 Hypothesis를 이루는 Weight, Bias, Cost의 값의 변화를 그래프로 확인하고 나아가 이 값들을 그래프로 표현하여 값의 변화를 확인합니다. 또한 학습이 진행되는 과정에서 H(x)함수가 어떻게 변화하는지 동영상으로 기록하여 확인합니다.

## 구현과정

먼저 임의의 2차원 데이터를 경향성이 있도록 설정합니다.  


```
mat_cov = [[1,0.9],[0.9,1]]
mu =  [4,3]
N = 400
X = np.random.multivariate_normal(mu, mat_cov, N)
```

Hypothesis를 다음과 같이 정의합니다.  


![hypothesis](https://user-images.githubusercontent.com/44831709/130807611-38f189db-a6fd-441d-8457-8109efc1715e.png)


데이터를 몇 번 학습할지, learning rate는 얼마로 할지 설정하고 임의의 베타값을 설정합니다.

```
N_iter = 1000
N = X.shape[0]
alpha = 0.05
beta_0 = np.random.random()
beta_1 = np.random.random()
x1 = X[:,0]
x2 = X[:,1]
rng = np.random.default_rng(2021)
```



Cost function은 아래와 같이 MSE(Mean Squared Error)로 설정합니다.   


![cost_function](https://user-images.githubusercontent.com/44831709/130806508-eae6ef66-e175-4f52-acbf-edba20e9aa6f.png)


데이터를 학습니다. history 에 베타값을 저장하고 나중에 그래프를 그릴 때 활용합니다.

```
history = []
history.append([beta_0,beta_1])
for i in range(N_iter):
    beta_1_next = beta_1 - alpha * (1/N) * np.sum(2*beta_1*np.square(x1) -2*x1*x2 + 2*x1*beta_0)
    beta_0_next = beta_0 - alpha* (1/N) * np.sum(2*beta_0 - 2*x2 + 2*x1*beta_1)
    beta_1, beta_0 = beta_1_next, beta_0_next
    history.append([beta_0,beta_1])

```



## 정보

https://wikidocs.net/21670
