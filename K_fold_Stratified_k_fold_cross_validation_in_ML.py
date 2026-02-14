{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNWTwirOANFYXBvs+VyCOEx",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/waghvaishnav/Model-Evaluation-Fine-Tuning-in-the-Machine-Learning-Toolkit/blob/main/K_fold_Stratified_k_fold_cross_validation_in_ML.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# K fold cross Validation"
      ],
      "metadata": {
        "id": "UYUe-_P48XTe"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## importing modules and datasets :"
      ],
      "metadata": {
        "id": "mn-eJytNMvMP"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8y3S5eiIK-Vz"
      },
      "outputs": [],
      "source": [
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "\n",
        "data = load_iris()\n",
        "\n",
        "x = data.data\n",
        "y = data.target\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Vh1V3GtNYNas"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# K fold cross validation"
      ],
      "metadata": {
        "id": "iE9tqGKzM5f_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import KFold\n",
        "\n",
        "kf = KFold(n_splits=5,shuffle=True,random_state=42)\n"
      ],
      "metadata": {
        "id": "9bnyQ-qTMH6s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "\n",
        "for train_index,test_index in kf.split(x,y):\n",
        "  x_train,x_test = x[train_index],x[test_index]\n",
        "  y_train,y_test = y[train_index],y[test_index]\n",
        "  print(Counter(y_test))\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MyS1GBodYUO9",
        "outputId": "ec93d39e-df4e-4115-ba16-dd0cf4ecdfed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Counter({np.int64(2): 11, np.int64(0): 10, np.int64(1): 9})\n",
            "Counter({np.int64(0): 13, np.int64(1): 10, np.int64(2): 7})\n",
            "Counter({np.int64(0): 12, np.int64(1): 10, np.int64(2): 8})\n",
            "Counter({np.int64(2): 12, np.int64(1): 10, np.int64(0): 8})\n",
            "Counter({np.int64(2): 12, np.int64(1): 11, np.int64(0): 7})\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "qgvaPyhGa18e"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Evaluate Logistic model"
      ],
      "metadata": {
        "id": "Yd03utm_NikK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "import numpy as np\n",
        "\n",
        "\n",
        "# importing cross validation score\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "log_model = LogisticRegression()\n",
        "\n",
        "log_score = cross_val_score(log_model,x,y,cv=kf)\n",
        "log_score\n",
        "\n",
        "# evaluating cross validate score\n",
        "log_score = np.average(log_score)\n",
        "log_score"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "utMbICx5NmMi",
        "outputId": "da3dcbda-31a8-4386-813f-883558032faa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.float64(0.9733333333333334)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# evaluating cross validate :\n",
        "\n",
        "from sklearn.model_selection import cross_validate\n",
        "\n",
        "cross_val = cross_validate(LogisticRegression(),x,y,cv=kf,scoring=[\"accuracy\"])\n",
        "cross_val"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jnfL6h0-OyXj",
        "outputId": "ba2a48ae-fa63-4dcd-f8de-3bc437eef9b3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'fit_time': array([0.1426723 , 0.04808092, 0.13214111, 0.04204154, 0.32269239]),\n",
              " 'score_time': array([0.00359702, 0.00359201, 0.00372195, 0.00401878, 0.00163698]),\n",
              " 'test_accuracy': array([1.        , 1.        , 0.93333333, 0.96666667, 0.96666667])}"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Evaluate Decision Tree"
      ],
      "metadata": {
        "id": "Ot35FH0mNezb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "import numpy as np\n",
        "\n",
        "\n",
        "# importing cross validation score\n",
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "dec_model = DecisionTreeClassifier()\n",
        "\n",
        "dec_score = cross_val_score(dec_model,x,y,cv=kf,scoring=\"accuracy\")   # accuracy is default parameter for scoring.\n",
        "dec_score\n",
        "\n",
        "# evaluating cross validate score\n",
        "dec_score = np.average(dec_score)\n",
        "dec_score"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3PiqYOQLNhvw",
        "outputId": "bd86f721-6824-4e9f-e5ae-2f7c782ecea2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.float64(0.9533333333333335)"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# evaluating cross validate :\n",
        "\n",
        "from sklearn.model_selection import cross_validate\n",
        "\n",
        "cross_val = cross_validate(LogisticRegression(),x,y,cv=kf,scoring=[\"accuracy\"])\n",
        "cross_val"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LfEL93rgRKHZ",
        "outputId": "9ca0c82e-fa40-4fdc-92c0-de55b9cdc9ed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'fit_time': array([0.02763295, 0.0219841 , 0.02236557, 0.01626086, 0.01984215]),\n",
              " 'score_time': array([0.00157905, 0.00175858, 0.00139046, 0.00143003, 0.00149894]),\n",
              " 'test_accuracy': array([1.        , 1.        , 0.93333333, 0.96666667, 0.96666667])}"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "6eiYMcVPRMW0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Stratified K fold cross validation\n",
        "importing dataset and modules"
      ],
      "metadata": {
        "id": "JBKYoYiAeKZx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.datasets import make_classification\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "x,y = make_classification(n_samples=1000,n_features=10,n_informative=8,weights=[0.9,0.10])\n",
        "\n",
        "# splits data into train_test_split\n",
        "x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)\n"
      ],
      "metadata": {
        "id": "oD5rG_yDeQpU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# percentage of class samples in each class\n",
        "\n",
        "from collections import Counter\n",
        "\n",
        "Counter(y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OBojU_GJeyGs",
        "outputId": "73a1f05d-6028-4985-86bb-70e22139b89f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Counter({np.int64(0): 175, np.int64(1): 25})"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import StratifiedKFold\n",
        "\n",
        "skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)\n",
        "\n",
        "for train_index,test_index in skf.split(x,y):\n",
        "  x_test,x_train = x[train_index],x[test_index]\n",
        "  y_train,y_test = y[train_index],y[test_index]"
      ],
      "metadata": {
        "id": "boMeCRR6fVKI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from collections import Counter\n",
        "Counter(y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "x-d_SOHDgcfH",
        "outputId": "6570ca33-8b13-4cd1-edad-061f3d5cb5b9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Counter({np.int64(1): 21, np.int64(0): 179})"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluating Cross_val_score for Logistic model"
      ],
      "metadata": {
        "id": "sXxjcerijZhS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "\n",
        "log_model = LogisticRegression()\n",
        "\n",
        "score = cross_val_score(log_model,x,y,cv=skf)\n",
        "\n",
        "np.average(score)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "grg-rSsGgdF4",
        "outputId": "a9318211-49c8-4647-9820-2b23408363a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.float64(0.9490000000000001)"
            ]
          },
          "metadata": {},
          "execution_count": 55
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# cross validate\n",
        "\n",
        "from sklearn.model_selection import cross_validate\n",
        "\n",
        "cross_validate(log_model,x,y,cv=skf,scoring=[\"accuracy\",\"precision\",\"recall\",\"roc_auc\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m2VhK8URkRfg",
        "outputId": "99c54ca5-b880-4c45-b0d1-5a826df8934b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'fit_time': array([0.0059166 , 0.0053091 , 0.00497031, 0.00477934, 0.00470996]),\n",
              " 'score_time': array([0.00905323, 0.00912952, 0.00875163, 0.00982428, 0.00931931]),\n",
              " 'test_accuracy': array([0.97 , 0.95 , 0.955, 0.94 , 0.93 ]),\n",
              " 'test_precision': array([0.88888889, 0.85714286, 0.875     , 0.90909091, 0.66666667]),\n",
              " 'test_recall': array([0.8       , 0.6       , 0.66666667, 0.47619048, 0.66666667]),\n",
              " 'test_roc_auc': array([0.98027778, 0.94694444, 0.96089385, 0.93907954, 0.87443469])}"
            ]
          },
          "metadata": {},
          "execution_count": 56
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "c_fmCvxclBQT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluating Cross_val_score for Decision Tree model"
      ],
      "metadata": {
        "id": "RRMPJGnXlLSr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "\n",
        "dec_model = DecisionTreeClassifier()\n",
        "\n",
        "# cross validation score\n",
        "score = cross_val_score(dec_model,x,y,cv=skf)\n",
        "\n",
        "np.average(score)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iFq743yPlOG8",
        "outputId": "95a69e07-7642-42fd-bf32-8454076b2e8f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.float64(0.931)"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# cross validate\n",
        "\n",
        "from sklearn.model_selection import cross_validate\n",
        "\n",
        "cross_validate(dec_model,x,y,cv=skf,scoring=[\"accuracy\",\"precision\",\"recall\",\"roc_auc\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G0n-is1WlR4j",
        "outputId": "728779ca-f682-4003-9154-332e91414ed2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'fit_time': array([0.01377511, 0.01411319, 0.01563454, 0.0109694 , 0.01193357]),\n",
              " 'score_time': array([0.00707769, 0.00665379, 0.00732684, 0.00683022, 0.00668001]),\n",
              " 'test_accuracy': array([0.94 , 0.945, 0.915, 0.93 , 0.93 ]),\n",
              " 'test_precision': array([0.7       , 0.76470588, 0.57692308, 0.66666667, 0.70588235]),\n",
              " 'test_recall': array([0.7       , 0.65      , 0.71428571, 0.66666667, 0.57142857]),\n",
              " 'test_roc_auc': array([0.83333333, 0.81388889, 0.8264166 , 0.81378026, 0.77174781])}"
            ]
          },
          "metadata": {},
          "execution_count": 58
        }
      ]
    }
  ]
}