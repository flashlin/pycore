# -*- coding: UTF-8 -*-
import re
from pypinyin import lazy_pinyin, Style


# from polyglot.text import Text, Word
# words = ["preprocessing", "processor", "invaluable", "thankful", "crossed"]
# for w in words:
#   w = Word(w, language="en")
#   print("{:<20}{}".format(w, w.morphemes))


def remove_special_symbol_generator(text):
    for word in text:
        if word in ['《', '》', '⋯', '；', '：', '～', '？', '！', '，', '、', '丶', '。', '「', '」', '?', '.',
                    '[', ']']:
            continue
        yield word


def replace_ch_text(text):
    chs = {
        'ㄟ': '欸',
        'Ａ': 'a', 'Ｂ': 'b', 'Ｃ': 'c', 'Ｄ': 'd', 'Ｅ': 'e',
        'Ｆ': 'f', 'Ｇ': 'g', 'Ｈ': 'h', 'Ｉ': 'i', 'Ｊ': 'j',
        'Ｋ': 'k', 'Ｌ': 'l', 'Ｍ': 'm', 'Ｎ': 'n', 'Ｏ': 'o',
        'Ｐ': 'p', 'Ｑ': 'q', 'Ｒ': 'r', 'Ｓ': 's', 'Ｔ': 't',
        'Ｕ': 'u', 'Ｖ': 'v', 'Ｗ': 'w', 'Ｘ': 'x', 'Ｙ': 'y',
        'Ｚ': 'x',
        '１': '1', '２': '2', '３': '3', '４': '4', '５': '5',
        '６': '6', '７': '7', '８': '8', '９': '9', '０': '0',
        'ｐ': 'p',
        'ｉ': 'i', 'ｎ': 'n', 'ｇ': 'g', 'ａ': 'a', 'ｔ': 't',
        '⽣': '生',
        '柺': '拐', '庄': '莊',
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '零': '0'
    }
    new_text = ""
    for ch in text:
        if ch in chs:
            new_text += chs[ch]
        else:
            new_text += ch
    return new_text


def remove_special_symbol_tolist(text):
    newTextList = []
    for word in remove_special_symbol_generator(text):
        newTextList.append(word)
    return newTextList


def remove_special_symbol(text):
    return ''.join(remove_special_symbol_generator(text))


def normal_text(text):
    text = text.lower()
    text = remove_special_symbol_generator(text)
    text = replace_ch_text(text)
    return text


def tongyong_to_list(tongyong_str):
    result = []
    m = re.findall('[A-Za-z]+\d', tongyong_str)
    for tongyong in m:
        result.append(tongyong)
    return result


def replace_word(word):
    def replace_tone(t, ch1, ch2):
        for n in range(1, 5):
            t = re.sub(f'^{ch1}{n}', f'{ch2}{n}', t)
        return t

    word = replace_tone(word, 'you', 'yo')
    word = replace_tone(word, 'n', 'en')
    return word


def chinese_to_pinyin(text):
    result = ""
    for word in lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True):
        # result += " " + replace_word(word)
        result += " " + word
    return result.strip()


def convert_to_pinyin(text):
    text = remove_special_symbol(text)
    return chinese_to_pinyin(text).strip()


def normal_hangyu(text):
    text_list = tongyong_to_list(text)
    result = ""
    for word in text_list:
        word = replace_word(word)
        result += f" {word}"
    return result


base_dict = {
    'ㄚ': ['a'],
    'ㄞ': ['ai'],
    'ㄢ': ['an'],
    'ㄤ': ['ang'],
    'ㄠ': ['ao'],
    'ㄅㄚ': ['ba'],
    'ㄅㄞ': ['bai'],
    'ㄅㄢ': ['ban'],
    'ㄅㄤ': ['bang'],
    'ㄅㄠ': ['bao'],
    'ㄅㄟ': ['bei'],
    'ㄅㄣ': ['ben'],
    'ㄅㄥ': ['beng'],
    'ㄅㄧ': ['bi'],
    'ㄅㄧㄢ': ['bian'],
    'ㄅㄧㄠ': ['biao'],
    'ㄅㄧㄝ': ['bie'],
    'ㄅㄧㄣ': ['bin'],
    'ㄅㄧㄥ': ['bing'],
    'ㄅㄛ': ['bo'],
    'ㄅㄨ': ['bu'],
    'ㄘㄚ': ['ca'],
    'ㄘㄞ': ['cai'],
    'ㄘㄢ': ['can'],
    'ㄘㄤ': ['cang'],
    'ㄘㄠ': ['cao'],
    'ㄘㄜ': ['ce'],
    'ㄘㄣ': ['cen'],
    'ㄘㄥ': ['ceng'],
    'ㄔㄚ': ['cha'],
    'ㄔㄞ': ['chai'],
    'ㄔㄢ': ['chan'],
    'ㄔㄤ': ['chang'],
    'ㄔㄠ': ['chao'],
    'ㄔㄜ': ['che'],
    'ㄔㄣ': ['chen'],
    'ㄔㄥ': ['cheng'],
    'ㄔ': ['chih', 'chi'],
    'ㄔㄨㄥ': ['chong'],
    'ㄔㄡ': ['chou'],
    'ㄔㄨ': ['chu'],
    'ㄔㄨㄚ': ['chua'],
    'ㄔㄨㄞ': ['chuai'],
    'ㄔㄨㄢ': ['chuan'],
    'ㄔㄨㄤ': ['chuang'],
    'ㄔㄨㄟ': ['chuei', 'chui'],
    'ㄔㄨㄣ': ['chun'],
    'ㄔㄨㄛ': ['chuo'],
    'ㄘ': ['cih', 'ci'],
    'ㄘㄨㄥ': ['cong'],
    'ㄘㄡ': ['cou'],
    'ㄘㄨ': ['cu'],
    'ㄘㄨㄢ': ['cuan'],
    'ㄘㄨㄟ': ['cuei', 'cui'],
    'ㄘㄨㄣ': ['cun'],
    'ㄘㄨㄛ': ['cuo'],
    'ㄉㄚ': ['da'],
    'ㄉㄞ': ['dai'],
    'ㄉㄢ': ['dan'],
    'ㄉㄤ': ['dang'],
    'ㄉㄠ': ['dao'],
    'ㄉㄜ': ['de'],
    'ㄉㄟ': ['dei'],
    'ㄉㄣ': ['den'],
    'ㄉㄥ': ['deng'],
    'ㄉㄧ': ['di'],
    'ㄉㄧㄢ': ['dian'],
    'ㄉㄧㄤ': ['diang'],
    'ㄉㄧㄠ': ['diao'],
    'ㄉㄧㄝ': ['die'],
    'ㄉㄧㄥ': ['ding'],
    'ㄉㄧㄡ': ['diou', 'diu'],
    'ㄉㄨㄥ': ['dong'],
    'ㄉㄡ': ['dou'],
    'ㄉㄨ': ['du'],
    'ㄉㄨㄢ': ['duan'],
    'ㄉㄨㄟ': ['duei', 'dui'],
    'ㄉㄨㄣ': ['dun'],
    'ㄉㄨㄛ': ['duo'],
    'ㄜ': ['e'],
    'ㄝ': ['e', 'ê'],
    'ㄟ': ['ei'],
    'ㄣ': ['en'],
    'ㄥ': ['ong', 'eng'],
    'ㄦ': ['er'],
    'ㄈㄚ': ['fa'],
    'ㄈㄢ': ['fan'],
    'ㄈㄤ': ['fang'],
    'ㄈㄟ': ['fei'],
    'ㄈㄣ': ['fen'],
    'ㄈㄥ': ['fong', 'feng'],
    'ㄈㄛ': ['fo'],
    'ㄈㄡ': ['fou'],
    'ㄈㄨ': ['fu'],
    'ㄍㄚ': ['ga'],
    'ㄍㄞ': ['gai'],
    'ㄍㄢ': ['gan'],
    'ㄍㄤ': ['gang'],
    'ㄍㄠ': ['gao'],
    'ㄍㄜ': ['ge'],
    'ㄍㄟ': ['gei'],
    'ㄍㄣ': ['gen'],
    'ㄍㄥ': ['geng'],
    'ㄍㄨㄥ': ['gong'],
    'ㄍㄡ': ['gou'],
    'ㄍㄨ': ['gu'],
    'ㄍㄨㄚ': ['gua'],
    'ㄍㄨㄞ': ['guai'],
    'ㄍㄨㄢ': ['guan'],
    'ㄍㄨㄤ': ['guang'],
    'ㄍㄨㄟ': ['guei', 'gui'],
    'ㄍㄨㄣ': ['gun'],
    'ㄍㄨㄛ': ['guo'],
    'ㄏㄚ': ['ha'],
    'ㄏㄞ': ['hai'],
    'ㄏㄢ': ['han'],
    'ㄏㄤ': ['hang'],
    'ㄏㄠ': ['hao'],
    'ㄏㄜ': ['he'],
    'ㄏㄟ': ['hei'],
    'ㄏㄣ': ['hen'],
    'ㄏㄥ': ['heng'],
    'ㄏㄨㄥ': ['hong'],
    'ㄏㄡ': ['hou'],
    'ㄏㄨ': ['hu'],
    'ㄏㄨㄚ': ['hua'],
    'ㄏㄨㄞ': ['huai'],
    'ㄏㄨㄢ': ['huan'],
    'ㄏㄨㄤ': ['huang'],
    'ㄏㄨㄟ': ['huei', 'hui'],
    'ㄏㄨㄣ': ['hun'],
    'ㄏㄨㄛ': ['huo'],
    'ㄐㄧ': ['ji'],
    'ㄐㄧㄚ': ['jia'],
    'ㄐㄧㄢ': ['jian'],
    'ㄐㄧㄤ': ['jiang'],
    'ㄐㄧㄠ': ['jiao'],
    'ㄐㄧㄝ': ['jie'],
    'ㄐㄧㄣ': ['jin'],
    'ㄐㄧㄥ': ['jing'],
    'ㄐㄩㄥ': ['jyong', 'jiong'],
    'ㄐㄧㄡ': ['jiou', 'jiu'],
    'ㄐㄩ': ['jyu', 'ju'],
    'ㄐㄩㄢ': ['jyuan', 'juan'],
    'ㄐㄩㄝ': ['jyue', 'jue'],
    'ㄐㄩㄣ': ['jyun', 'jun'],
    'ㄎㄚ': ['ka'],
    'ㄎㄞ': ['kai'],
    'ㄎㄢ': ['kan'],
    'ㄎㄤ': ['kang'],
    'ㄎㄠ': ['kao'],
    'ㄎㄜ': ['ke'],
    'ㄎㄟ': ['kei'],
    'ㄎㄣ': ['ken'],
    'ㄎㄥ': ['keng'],
    'ㄎㄨㄥ': ['kong'],
    'ㄎㄡ': ['kou'],
    'ㄎㄨ': ['ku'],
    'ㄎㄨㄚ': ['kua'],
    'ㄎㄨㄞ': ['kuai'],
    'ㄎㄨㄢ': ['kuan'],
    'ㄎㄨㄤ': ['kuang'],
    'ㄎㄨㄟ': ['kuei', 'kui'],
    'ㄎㄨㄣ': ['kun'],
    'ㄎㄨㄛ': ['kuo'],
    'ㄌㄚ': ['la'],
    'ㄌㄞ': ['lai'],
    'ㄌㄢ': ['lan'],
    'ㄌㄤ': ['lang'],
    'ㄌㄠ': ['lao'],
    'ㄌㄜ': ['le'],
    'ㄌㄟ': ['lei'],
    'ㄌㄥ': ['leng'],
    'ㄌㄧ': ['li'],
    'ㄌㄧㄚ': ['lia'],
    'ㄌㄧㄢ': ['lian'],
    'ㄌㄧㄤ': ['liang'],
    'ㄌㄧㄠ': ['liao'],
    'ㄌㄧㄝ': ['lie'],
    'ㄌㄧㄣ': ['lin'],
    'ㄌㄧㄥ': ['ling'],
    'ㄌㄧㄡ': ['liou', 'liu'],
    'ㄌㄛ': ['lo'],
    'ㄌㄨㄥ': ['long'],
    'ㄌㄡ': ['lou'],
    'ㄌㄨ': ['lu'],
    'ㄌㄩ': ['lyu', 'lv'],
    'ㄌㄨㄢ': ['luan'],
    'ㄌㄩㄢ': ['lyuan', 'lvan'],
    'ㄌㄩㄝ': ['lyue', 'lve'],
    'ㄌㄨㄣ': ['lun'],
    'ㄌㄩㄣ': ['lyun', 'lvn'],
    'ㄌㄨㄛ': ['luo'],
    'ㄇㄚ': ['ma'],
    'ㄇㄞ': ['mai'],
    'ㄇㄢ': ['man'],
    'ㄇㄤ': ['mang'],
    'ㄇㄠ': ['mao'],
    'ㄇㄜ': ['me'],
    'ㄇㄟ': ['mei'],
    'ㄇㄣ': ['men'],
    'ㄇㄥ': ['meng'],
    'ㄇㄧ': ['mi'],
    'ㄇㄧㄢ': ['mian'],
    'ㄇㄧㄠ': ['miao'],
    'ㄇㄧㄝ': ['mie'],
    'ㄇㄧㄣ': ['min'],
    'ㄇㄧㄥ': ['ming'],
    'ㄇㄧㄡ': ['miou', 'miu'],
    'ㄇㄛ': ['mo'],
    'ㄇㄡ': ['mou'],
    'ㄇㄨ': ['mu'],
    'ㄋㄚ': ['na'],
    'ㄋㄞ': ['nai'],
    'ㄋㄢ': ['nan'],
    'ㄋㄤ': ['nang'],
    'ㄋㄠ': ['nao'],
    'ㄋㄜ': ['ne'],
    'ㄋㄟ': ['nei'],
    'ㄋㄣ': ['nen'],
    'ㄋㄥ': ['neng'],
    'ㄋㄧ': ['ni'],
    'ㄋㄧㄚ': ['nia'],
    'ㄋㄧㄢ': ['nian'],
    'ㄋㄧㄤ': ['niang'],
    'ㄋㄧㄠ': ['niao'],
    'ㄋㄧㄝ': ['nie'],
    'ㄋㄧㄣ': ['nin'],
    'ㄋㄧㄥ': ['ning'],
    'ㄋㄧㄡ': ['niou', 'niu'],
    'ㄋㄨㄥ': ['nong'],
    'ㄋㄡ': ['nou'],
    'ㄋㄨ': ['nu'],
    'ㄋㄩ': ['nyu', 'nv'],
    'ㄋㄨㄢ': ['nuan'],
    'ㄋㄩㄝ': ['nyue', 'nve'],
    'ㄋㄨㄣ': ['nun'],
    'ㄋㄨㄛ': ['nuo'],
    'ㄛ': ['o'],
    'ㄡ': ['ou'],
    'ㄆㄚ': ['pa'],
    'ㄆㄞ': ['pai'],
    'ㄆㄢ': ['pan'],
    'ㄆㄤ': ['pang'],
    'ㄆㄠ': ['pao'],
    'ㄆㄟ': ['pei'],
    'ㄆㄣ': ['pen'],
    'ㄆㄥ': ['peng'],
    'ㄆㄧ': ['pi'],
    'ㄆㄧㄢ': ['pian'],
    'ㄆㄧㄠ': ['piao'],
    'ㄆㄧㄝ': ['pie'],
    'ㄆㄧㄣ': ['pin'],
    'ㄆㄧㄥ': ['ping'],
    'ㄆㄛ': ['po'],
    'ㄆㄡ': ['pou'],
    'ㄆㄨ': ['pu'],
    'ㄑㄧ': ['ci', 'qi'],
    'ㄑㄧㄚ': ['cia', 'qia'],
    'ㄑㄧㄢ': ['cian', 'qian'],
    'ㄑㄧㄤ': ['ciang', 'qiang'],
    'ㄑㄧㄠ': ['ciao', 'qiao'],
    'ㄑㄧㄝ': ['cie', 'qie'],
    'ㄑㄧㄣ': ['cin', 'qin'],
    'ㄑㄧㄥ': ['cing', 'qing'],
    'ㄑㄩㄥ': ['cyong', 'qiong'],
    'ㄑㄧㄡ': ['ciou', 'qiu'],
    'ㄑㄩ': ['cyu', 'qu'],
    'ㄑㄩㄢ': ['cyuan', 'quan'],
    'ㄑㄩㄝ': ['cyue', 'que'],
    'ㄑㄩㄣ': ['cyun', 'qun'],
    'ㄖㄢ': ['ran'],
    'ㄖㄤ': ['rang'],
    'ㄖㄠ': ['rao'],
    'ㄖㄜ': ['re'],
    'ㄖㄣ': ['ren'],
    'ㄖㄥ': ['reng'],
    'ㄖ': ['rih', 'ri'],
    'ㄖㄨㄥ': ['rong'],
    'ㄖㄡ': ['rou'],
    'ㄖㄨ': ['ru'],
    'ㄖㄨㄢ': ['ruan'],
    'ㄖㄨㄟ': ['ruei', 'rui'],
    'ㄖㄨㄣ': ['run'],
    'ㄖㄨㄛ': ['ruo'],
    'ㄙㄚ': ['sa'],
    'ㄙㄞ': ['sai'],
    'ㄙㄢ': ['san'],
    'ㄙㄤ': ['sang'],
    'ㄙㄠ': ['sao'],
    'ㄙㄜ': ['se'],
    'ㄙㄟ': ['sei'],
    'ㄙㄣ': ['sen'],
    'ㄙㄥ': ['seng'],
    'ㄕㄚ': ['sha'],
    'ㄕㄞ': ['shai'],
    'ㄕㄢ': ['shan'],
    'ㄕㄤ': ['shang'],
    'ㄕㄠ': ['shao'],
    'ㄕㄜ': ['she'],
    'ㄕㄟ': ['shei'],
    'ㄕㄣ': ['shen'],
    'ㄕㄥ': ['sheng'],
    'ㄕ': ['shih', 'shi'],
    'ㄕㄨㄥ': ['shong'],
    'ㄕㄡ': ['shou'],
    'ㄕㄨ': ['shu'],
    'ㄕㄨㄚ': ['shua'],
    'ㄕㄨㄞ': ['shuai'],
    'ㄕㄨㄢ': ['shuan'],
    'ㄕㄨㄤ': ['shuang'],
    'ㄕㄨㄟ': ['shuei', 'shui'],
    'ㄕㄨㄣ': ['shun'],
    'ㄕㄨㄛ': ['shuo'],
    'ㄙ': ['sih', 'si'],
    'ㄙㄨㄥ': ['song'],
    'ㄙㄡ': ['sou'],
    'ㄙㄨ': ['su'],
    'ㄙㄨㄢ': ['suan'],
    'ㄙㄨㄟ': ['suei', 'sui'],
    'ㄙㄨㄣ': ['sun'],
    'ㄙㄨㄛ': ['suo'],
    'ㄊㄚ': ['ta'],
    'ㄊㄞ': ['tai'],
    'ㄊㄢ': ['tan'],
    'ㄊㄤ': ['tang'],
    'ㄊㄠ': ['tao'],
    'ㄊㄜ': ['te'],
    'ㄊㄟ': ['tei'],
    'ㄊㄥ': ['teng'],
    'ㄊㄧ': ['ti'],
    'ㄊㄧㄢ': ['tian'],
    'ㄊㄧㄠ': ['tiao'],
    'ㄊㄧㄝ': ['tie'],
    'ㄊㄧㄥ': ['ting'],
    'ㄊㄨㄥ': ['tong'],
    'ㄊㄡ': ['tou'],
    'ㄊㄨ': ['tu'],
    'ㄊㄨㄢ': ['tuan'],
    'ㄊㄨㄟ': ['tuei', 'tui'],
    'ㄊㄨㄣ': ['tun'],
    'ㄊㄨㄛ': ['tuo'],
    'ㄨㄚ': ['wa'],
    'ㄨㄞ': ['wai'],
    'ㄨㄢ': ['wan'],
    'ㄨㄤ': ['wang'],
    'ㄨㄟ': ['wei'],
    'ㄨㄣ': ['wun', 'wen'],
    'ㄨㄥ': ['wong', 'weng'],
    'ㄨㄛ': ['wo'],
    'ㄨ': ['wu'],
    'ㄒㄧ': ['si', 'xi'],
    'ㄒㄧㄚ': ['sia', 'xia'],
    'ㄒㄧㄢ': ['sian', 'xian'],
    'ㄒㄧㄤ': ['siang', 'xiang'],
    'ㄒㄧㄠ': ['siao', 'xiao'],
    'ㄒㄧㄝ': ['sie', 'xie'],
    'ㄒㄧㄣ': ['sin', 'xin'],
    'ㄒㄧㄥ': ['sing', 'xing'],
    'ㄒㄩㄥ': ['syong', 'xiong'],
    'ㄒㄧㄡ': ['siou', 'xiu'],
    'ㄒㄩ': ['syu', 'xu'],
    'ㄒㄩㄢ': ['syuan', 'xuan'],
    'ㄒㄩㄝ': ['syue', 'xue'],
    'ㄒㄩㄣ': ['syun', 'xun'],
    'ㄧㄚ': ['ya'],
    'ㄧㄢ': ['yan'],
    'ㄧㄤ': ['yang'],
    'ㄧㄠ': ['yao'],
    'ㄧㄝ': ['ye'],
    'ㄧ': ['yi'],
    'ㄧㄣ': ['yin'],
    'ㄧㄥ': ['ying'],
    'ㄩㄥ': ['yong'],
    'ㄧㄡ': ['you', 'yo'],
    'ㄩ': ['yu'],
    'ㄩㄢ': ['yuan'],
    'ㄩㄝ': ['yue'],
    'ㄩㄣ': ['yun'],
    'ㄗㄚ': ['za'],
    'ㄗㄞ': ['zai'],
    'ㄗㄢ': ['zan'],
    'ㄗㄤ': ['zang'],
    'ㄗㄠ': ['zao'],
    'ㄗㄜ': ['ze'],
    'ㄗㄟ': ['zei'],
    'ㄗㄣ': ['zen'],
    'ㄗㄥ': ['zeng'],
    'ㄓㄚ': ['jha', 'zha'],
    'ㄓㄞ': ['jhai', 'zhai'],
    'ㄓㄢ': ['jhan', 'zhan'],
    'ㄓㄤ': ['jhang', 'zhang'],
    'ㄓㄠ': ['jhao', 'zhao'],
    'ㄓㄜ': ['jhe', 'zhe'],
    'ㄓㄟ': ['jhei', 'zhei'],
    'ㄓㄣ': ['jhen', 'zhen'],
    'ㄓㄥ': ['jheng', 'zheng'],
    'ㄓ': ['jhih', 'zhi'],
    'ㄓㄨㄥ': ['jhong', 'zhong'],
    'ㄓㄡ': ['jhou', 'zhou'],
    'ㄓㄨ': ['jhu', 'zhu'],
    'ㄓㄨㄚ': ['jhua', 'zhua'],
    'ㄓㄨㄞ': ['jhuai', 'zhuai'],
    'ㄓㄨㄢ': ['jhuan', 'zhuan'],
    'ㄓㄨㄤ': ['jhuang', 'zhuang'],
    'ㄓㄨㄟ': ['jhuei', 'zhui'],
    'ㄓㄨㄣ': ['jhun', 'zhun'],
    'ㄓㄨㄛ': ['jhuo', 'zhuo'],
    'ㄗ': ['zih', 'zi'],
    'ㄗㄨㄥ': ['zong'],
    'ㄗㄡ': ['zou'],
    'ㄗㄨ': ['zu'],
    'ㄗㄨㄢ': ['zuan'],
    'ㄗㄨㄟ': ['zuei', 'zui'],
    'ㄗㄨㄣ': ['zun'],
    'ㄗㄨㄛ': ['zuo']
}

tone_list = ['', 'ˊ', 'ˋ', 'ˇ', '˙']


def get_idx_tone_list():
    return zip(range(len(tone_list)), tone_list)


def create_tongyong_phonetic_dict_by_tone(dict, tone_idx, tone_ch):
    for phonetic_key in base_dict.keys():
        tongyong_val = base_dict[phonetic_key][0]
        dict[f"{tongyong_val}{tone_idx + 1}"] = f"{phonetic_key}{tone_ch}"


def create_tongyong_phonetic_dict():
    dict = {}
    for tone_idx, tone_ch in get_idx_tone_list():
        create_tongyong_phonetic_dict_by_tone(dict, tone_idx, tone_ch)
    return dict


tongyong_phonetic_dict = create_tongyong_phonetic_dict()


def create_hangyu_phonetic_dict_by_tone(dict, tone_idx, tone_ch):
    for phonetic_key in base_dict.keys():
        vals = base_dict[phonetic_key]
        val = vals[0]
        if len(vals) > 1:
            val = vals[1]
        dict[f"{val}{tone_idx + 1}"] = f"{phonetic_key}{tone_ch}"


def create_hangyu_phonetic_dict():
    dict = {}
    for tone_idx, tone_ch in get_idx_tone_list():
        create_hangyu_phonetic_dict_by_tone(dict, tone_idx, tone_ch)
    return dict


hangyu_phonetic_dict = create_hangyu_phonetic_dict()


def create_tongyong_phonetic_dict():
    dict = {}
    for tongyong_key in tongyong_phonetic_dict.keys():
        phonetic_val = tongyong_phonetic_dict[tongyong_key][0]
        dict[phonetic_val] = tongyong_key
    return dict


phonetic_tongyong_dict = create_tongyong_phonetic_dict()


def create_hangyu_phonetic_dict():
    dict = {}
    for key in hangyu_phonetic_dict.keys():
        val = hangyu_phonetic_dict[key]
        dict[val] = key
    return dict


phonetic_hangyu_dict = create_hangyu_phonetic_dict()


def zip_dict(dict):
    return zip(range(len(dict)), dict.keys())


def create_tongyong_index_dict():
    dict = {}
    for idx, key in zip_dict(tongyong_phonetic_dict):
        dict[key] = idx
    return dict


tongyong_index_dict = create_tongyong_index_dict()


def create_phonetic_index_dict():
    dict = {}
    for idx, key in zip_dict(phonetic_tongyong_dict):
        dict[key] = idx
    return dict


phonetic_index_dict = create_phonetic_index_dict()


def create_hangyu_index_dict():
    dict = {}
    for idx, key in zip_dict(phonetic_hangyu_dict):
        hangyu_ch = phonetic_hangyu_dict[key]
        dict[hangyu_ch] = idx
    return dict


hangyu_index_dict = create_hangyu_index_dict()


def create_index_phonetic_dict():
    dict = {}
    for key in phonetic_index_dict.keys():
        index = phonetic_index_dict[key]
        dict[index] = key
    return dict


index_phonetic_dict = create_index_phonetic_dict()


def create_index_tongyong_dict():
    dict = {}
    for key in tongyong_index_dict.keys():
        index = tongyong_index_dict[key]
        dict[index] = key
    return dict


index_tongyong_dict = create_index_tongyong_dict()


def tongyong_to_bytes(tongyong_str):
    result = []
    m = re.findall('[A-Za-z]+\d', tongyong_str)
    for tongyong in m:
        index = tongyong_index_dict[tongyong]
        result.append(index)
    return result


def hangyu_to_bytes(hangyu_str):
    result = []
    m = re.findall('[A-Za-z]+\d', hangyu_str)
    for txt in m:
        index = hangyu_index_dict[txt]
        result.append(index)
    return result


def bytes_to_phonetic(bytes_list):
    result = []
    for index in bytes_list:
        phonetic = index_phonetic_dict[index]
        result.append(phonetic)
    return result


kk_list = [
    'i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'o', 'ɔ', 'u', 'ʌ', 'ə', 'ɪr', 'ɚ', 'ɝ',
    'aɪ', 'aᴜ', 'ɔɪ',
    'p', 't', 'k', 'f', 's', 'θ', 'ʃ', 'tʃ', 'h',
    'b', 'd', 'g', 'v', 'z', 'ð', 'ʒ', 'dʒ', 'm', 'n', 'ŋ', 'l', 'r', 'j', 'w',
    '-'
]


def remove_nonwords(text):
    return re.sub(r'[^A-Za-z12345]', "", text)


phonetic_vocab_size = len(phonetic_index_dict)

import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def target_encode_multiclass(X, y):  # X,y are pandas df and series
    y = y.astype(str)  # convert to string to onehot encode
    enc = ce.OneHotEncoder().fit(y)
    y_onehot = enc.transform(y)
    class_names = y_onehot.columns  # names of onehot encoded columns
    X_obj = X.select_dtypes('object')  # separate categorical columns
    X = X.select_dtypes(exclude='object')
    for class_ in class_names:
        enc = ce.TargetEncoder()
        enc.fit(X_obj, y_onehot[class_])  # convert all categorical
        temp = enc.transform(X_obj)  # columns for class_
        temp.columns = [str(x) + '_' + str(class_) for x in temp.columns]
        X = pd.concat([X, temp], axis=1)  # add to original dataset
    return X


from common.csv_utils import CsvReader, label_encode2, read_csv, target_encode_multiclass

if __name__ == "__main__":
    # print(phonetic_tongyong_dict)
    # print(tongyong_phonetic_dict)
    # print(tongyong_index_dict)
    # print(phonetic_index_dict)
    # print(index_phonetic_dict)
    # print(index_tongyong_dict)
    # train = pd.read_csv('test.csv', sep='\t', lineterminator='\r')
    # train = read_csv("test.csv")
    # train = label_encode2(train, ["type"])
    # train = target_encode_multiclass(train, train["type"])
    # train.info()
    # print(train)
    # te = ce.TargetEncoder(cols=train.columns.values, smoothing=0.3).fit(train, target)
    # train = te.transform(train)
    # print(hangyu_phonetic_dict)
    print(hangyu_index_dict)


def is_mandarin(ch):
    return re.match('^[\u4e00-\u9fa5]+$', ch)


def merge_eng_number_in_list(lst):
    eng = ""
    new_list = []
    for ch in lst:
        if ch.startswith('#'):
            eng += ch[1:]
        else:
            if eng != "":
                new_list.append(f"#{eng}")
                eng = ""
            new_list.append(ch)
    if eng != "":
        new_list.append(f"#{eng}")
    return new_list


def split_eng_number_space_in_list(lst):
    new_list = []
    for ch in lst:
        if ch.startswith('#'):
            ss = ch[1:].split(' ')
            new_list += [f"#{x}" for x in ss]
        else:
            new_list.append(ch)
    return new_list


def mandarin_english_to_list(text):
    text = normal_text(text)
    lst = []
    for ch in text:
        if is_mandarin(ch):
            lst.append(chinese_to_pinyin(ch))
        else:
            lst.append(f"#{ch}")
    lst = merge_eng_number_in_list(lst)
    lst = split_eng_number_space_in_list(lst)
    return lst


def split_eng_number1_space_in_list(lst):
    new_list = []
    for ch in lst:
        if ch.startswith('#'):
            ss = ch[1:]
            new_list += [f"#{x}" for x in ss]
        else:
            new_list.append(ch)
    return new_list


def mandarin_english1_to_list(text):
    text = normal_text(text)
    lst = []
    for ch in text:
        if is_mandarin(ch):
            lst.append(chinese_to_pinyin(ch))
        else:
            lst.append(f"#{ch}")
    lst = merge_eng_number_in_list(lst)
    lst = split_eng_number1_space_in_list(lst)
    return lst


def mandarin_english1_list_to_prepare_list(text_lst):
    lst = []
    for ch in text_lst:
        if ch.startswith('#'):
            lst.append(ch[1:])
        else:
            lst.append(ch)
    return lst


class PinyinVocabulary:
    def __init__(self, max_length=150):
        self.max_length = max_length
        self.char_to_idx = {}
        idx = 0
        eng = "abcdefghijklmnopqrstuvwxyz1234567890 "
        for ch in eng:
            self.char_to_idx[ch] = idx
            idx += 1
        with open("./pinyin.txt", "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                self.char_to_idx[line] = idx
                idx += 1
        self.idx_to_char = {}
        for key in self.char_to_idx.keys():
            idx = self.char_to_idx[key]
            self.idx_to_char[idx] = key

    def __call__(self, text):
        text_lst = mandarin_english1_to_list(text)
        text_lst = mandarin_english1_list_to_prepare_list(text_lst)
        text_lst_len = len(text_lst)
        if text_lst_len > self.max_length:
            raise ValueError(f"except {self.max_length} len, but got {len(text_lst)}")
        return [self.char_to_idx[ch] for ch in text_lst]

    def pad_text(self, text):
        text_lst = self(text)
        pad_len = self.max_len - len(text_lst)
        pad_lst = text_lst + [0] * pad_len
        return pad_lst

    def get_vocabulary(self):
        return self.char_to_idx

    def get_size(self):
        return len(self.char_to_idx.keys())
