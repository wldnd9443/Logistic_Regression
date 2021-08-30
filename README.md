# # Logistic Regression
> Logistic Regression 구현해보기

앞서 배웠던 Linear Regression과 달리 Logistic Regression은 어떤 방법으로 학습하는지 그래프와 그림을 통해 확인합니다.


![](../header.png)

## 소개

  이 Repository에서는 Logistic Regression을 구현해 볼 것입니다. 이전에 다뤘던 Linear Regression과 마찬가지로 Logistic Regression도 기존에 주어진 데이터를 대표할 수 있는 그래프를 찾아야 합니다. 그러나 Linear Regression과 달리 Logistic Regression은 데이터를 True와 False로 구분하는 classification 입니다. 학습을 완료하고 입력에 따른 결과값이 Linear와 다르게 True 나 False를  반환해야 합니다.    
  

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
