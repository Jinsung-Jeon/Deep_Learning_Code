# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:49:46 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap5-Classification Flower')
from dataset import *


# 푸시다운오토마타 알파벳의 지정
MIN_LENGTH = 10
MAX_LENGTH = 40

ALPHA = [chr(n) for n in range(ord('a'), ord('z')+1)]
DIGIT = [chr(n) for n in range(ord('0'), ord('9')+1)]

EOS = ['$']
ADDOP = ['+', '-']
MULTOP = ['*', '/']
LPAREN = ['(']
RPAREN = [')']

SYMBOLS = EOS + ADDOP + MULTOP + LPAREN + RPAREN
ALPHANUM = ALPHA + DIGIT
ALPHABET = SYMBOLS + ALPHANUM

# 푸시다운오토마타 문법 규칙의 지정
S = 0   #SENT
E = 1   #EXP
T = 2   #TERM
F = 3   #FACTOR
V = 4   #VARIABLE
N = 5   #NUMVER
V2 = 6  #VAR_TAIL

RULES = {
        S: [[E]],
        E: [[T], [E, ADDOP, T]],
        T: [[F], [T, MULTOP, F]],
        F: [[V], [N], [LPAREN, E, RPAREN]],
        V: [[ALPHA], [ALPHA, V2]],
        V2: [[ALPHANUM], [ALPHANUM, V2]],
        N: [[DIGIT], [DIGIT, N]]
        }

# 푸시다운오토마타 파싱 테이블의 지정
E_NEXT = EOS + RPAREN + ADDOP
T_NEXT = E_NEXT + MULTOP
F_NEXT = T_NEXT
V_NEXT = F_NEXT
N_NEXT = F_NEXT

action_table = {
    0: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    1: [[ADDOP, 0], [EOS, 0]],
    2: [[MULTOP, 10], [E_NEXT, -1, E]],
    3: [[T_NEXT, -1, T]],
    4: [[F_NEXT, -1, F]],
    5: [[F_NEXT, -1, F]],
    6: [[ALPHANUM, 6], [V_NEXT, -1, V]],
    7: [[DIGIT, 7], [N_NEXT, -1, N]],
    8: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    9: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    10: [[ALPHA, 6], [DIGIT, 7], [LPAREN, 8]],
    11: [[V_NEXT, -2, V]],
    12: [[N_NEXT, -2, N]],
    13: [[RPAREN, 16], [ADDOP, 9]],
    14: [[MULTOP, 10], [T_NEXT, -3, T]],
    15: [[F_NEXT, -3, F]],
    16: [[F_NEXT, -3, F]],
    }

goto_table = {
    0: { E:1, T:2, F:3, V:4, N:5},
    6: { V:11},
    7: { N:12},
    8: { E:13, T:2, F:3, V:4, N:5},
    9: { T:14, F:3, V:4, N:5},
    10: { F:15, V:4, N:5},
    }

# 클래스 선언과 객체 초기화 메서드 및 속성 메서드 재정의
class AutomataDataset(Dataset):
    def __init__(self):
        super(AutomataDataset, self).__init__('automata', 'binary')
        self.input_shape = [MAX_LENGTH+1, len(ALPHABET)]
        self.output_shape = [1]
        
    @property
    def train_count(self):
        return 10000
    
# 학습, 검증, 평가, 시각화 단계에 데이터를 공급하는 네 가지 메서드 재정의
def automata_get_train_data(self, batch_size, nth):
    return automata_generate_data(batch_size)

def automata_get_validate_data(self, count):
    return automata_generate_data(count)

def automata_get_test_data(self):
    return automata_get_test_data(1000)

def automata_generate_data(count):
    xs = np.zeros([count, MAX_LENGTH, len(ALPHABET)])
    ys = np.zeros([count, 1])
    
    for n in range(count):
        is_correct = n % 2
        
        if is_correct:
            sent = automata_generate_sent()
        else:
            while True:
                sent = automata_generate_sent()
                touch = np.random.randint(1, len(sent)//5)
                for k in range(touch):
                    sent_pos = np.random.randint(len(sent))
                    char_pos = np.random.randint(len(ALPHABET)-1)
                    sent = sent[:sent_pos] + ALPHABET[char_pos] + sent[sent_pos+1:]
                if not automata_is_correct_sent(sent):
                    break
        ords = [ALPHABET.index(ch) for ch in sent]
        xs[n, 0, 0] = len(sent)
        xs[n, 1:len(sent)+1, :] = np.eye(len(ALPHABET))[ords]
        ys[n, 0] = is_correct
        
    return xs, ys

AutomataDataset.get_train_data = automata_get_train_data
AutomataDataset.get_validate_data = automata_get_validate_data
AutomataDataset.get_test_data = automata_get_test_data
AutomataDataset.get_visualize_data = automata_get_validate_data

# 문장 생성 함수 정의
def automata_generate_sent():
    while True:
        try:
            sent = automata_gen_node(S, 0)
            if len(sent) >= MAX_LENGTH: continue
            if len(sent) <= MIN_LENGTH: continue
            return sent
        except Exception:
            continue

def automata_gen_node(node, depth):
    if depth > 30: raise Exception
    if node not in RULES: assert 0
    rules = RULES[node]
    nth = np.random.randint(len(rules))
    sent = ''
    for term in rules[nth]:
        if isinstance(term, list):
            pos = np.random.randint(len(term))
            sent += term[pos]
        else:
            sent += automata_gen_node(term, depth+1)
    return sent

# 문장 검사 함수 정의
def automata_is_correct_sent(sent):
    sent = sent + '$'
    states, pos, nextch = [0], 0, sent[0]
    
    while True:
        actions = action_table[states[-1]]
        found = False
        for pair in actions:
            if nextch not in pair[0]:
                continue
            found = True
            if pair[1] == 0:
                return True
            elif pair[1] > 0:
                states.append(pair[1])
                pos += 1
                nextch = sent[pos]
                break
            else:
                states = states[:pair[1]]
                goto = goto_table[states[-1]]
                goto_state = goto[pair[2]]
                states.append(goto_state)
                break
        if not found:
            return False

# 오토마타 데이터셋을 위한 시각화 메서드 정의하기
def automata_visualize(self, xs, est, ans):
    for n in range(len(xs)):
        length = int(xs[n, 0, 0])
        sent = np.argmax(xs[n, 1:length+1], axis = 1)
        text = "".join([ALPHABET[letter] for letter in sent])
        
        answer, guess, result = '잘못된 패턴', '탈락추정', 'X'
        
        if ans[n][0] > 0.5:
            answer = '올바른 패턴'
        if est[n][0] > 0.5:
            guess = '합격추정'
        if ans[n][0] > 0.5 and est[n][0] > 0.5:
            result = '0'
        if ans[n][0] < 0.5 and est[n][0] < 0.5:
            result = '0'
            
        print('{}: {} => {}({:4.2f}) : {}'.format(text, answer, guess, est[n][0], result))
        
AutomataDataset.visualize = automata_visualize

