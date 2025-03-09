
import os
import re
from turtle import pos
import msgpack
import json

import numpy as np
from tqdm import tqdm 


class CardType(object):
    Koikatu = "【KoiKatuChara】"
    KoikatsuParty = "【KoiKatuCharaS】"
    KoikatsuPartySpecialPatch ="【KoiKatuCharaSP】"
    EmotionCreators =  "【EroMakeChara】"
    AiSyoujyo = "【AIS_Chara】"
    KoikatsuSunshine = "【KoiKatuCharaSun】"
    RoomGirl = "【RG_Chara】"
    Types = [Koikatu, KoikatsuParty,KoikatsuSunshine, EmotionCreators, AiSyoujyo, RoomGirl ]


class CharacterCard(object):
    pngStartChunck:bytes = b"\x89\x50\x4E\x47\x0D"
    pngEndChunck:bytes = b"\x49\x45\x4E\x44\xAE\x42\x60\x82"
    cardData:bytes = None
    cardType:str = None
    data:dict = None
    def __init__(self,filePath=None):
        self.filePath = filePath
        if self.filePath is not None:
            self.parse(self.filePath)



    def parse(self, filePath):
        self.filePath = filePath if filePath is None else self.filePath
        if self.filePath is None:
            raise ValueError("filePath cannot be None")
        self.cardData=self.readCard(filePath)
        data=self.parseCard(self.cardData)
        self.data =None
        if data is not None:
            self.data=dict(data)

        return self.data
        

    def readCard(self, filePath=None)->bytes:
        self.filePath = filePath if filePath is None else self.filePath
        if self.filePath is None:
            raise ValueError("filePath cannot be None")
        
        with open(self.filePath,"rb") as f:
            cardData = f.read()

        return cardData

    def parseCard(self,cardData:bytes)->dict:
        cardType = self.getCardType(cardData)
        self.cardType = cardType
        data = None
        if cardType == CardType.AiSyoujyo:
            data = self.parseAiSyoujyo(cardData)
        elif cardType == CardType.KoikatsuSunshine:
            data = self.parseKoikatsuSunshine(cardData)
        else:
            print(f"connot supported card {self.filePath}")
            # raise ValueError(f"connot supported card type: {cardType}")
        
        return data

    def getCardType(self,cardData:bytes)->str:
        pos = cardData.find(self.pngEndChunck) + 8
        size = int.from_bytes(cardData[pos:pos+4], byteorder="little")
        pos += 4
        data = cardData[pos:pos+size]

        cardType = None
        for type in CardType.Types:
            if data.find(bytes(type,encoding="utf-8")) != -1:
                cardType = type
                break
            else:
                continue
        
        # if cardType is None:
        #     raise ValueError("Invalid card type")

        return cardType
    
    def parseAiSyoujyo(self,cardData:bytes)->dict:
        pos = cardData.find(self.pngEndChunck) + 8
        size = int.from_bytes(cardData[pos:pos+4], byteorder="little")
        pos += 4 + size
        lstInfoSize = int.from_bytes(cardData[pos:pos+4], byteorder="little") 
        pos += 4
        lstInfo:dict = msgpack.loads(cardData[pos:pos+lstInfoSize])
        pos += lstInfoSize
        dataSize = int.from_bytes(cardData[pos:pos+8], byteorder="little")
        dataPos = pos + 8
        data = cardData[dataPos:dataPos+dataSize]

        d = {}
        _pos = 0
        for item in lstInfo["lstInfo"]:
            name  = item["name"]
            if name == "KKEx":
                continue
            elif name in ["Custom","Coordinate"]:
                _pos =item["pos"]
                size = item["size"]
                _data = data[_pos:_pos+size]
                _pos_ = 0
                l = []
                while _pos_ < size:
                    _size = int.from_bytes(_data[_pos_:_pos_+4], byteorder="little")
                    _pos_ += 4
                    _data_ = msgpack.unpackb(_data[_pos_:_pos_+_size], strict_map_key=False)
                    _pos_ += _size
                    l.append(_data_)
                _data = l
                
            else:
                _pos = item["pos"]
                size = item["size"]
                _data = msgpack.unpackb(data[_pos:_pos+size], strict_map_key=False)
            d[name] = _data
            # _pos += size



        # d.update(lstInfo)
        return d


        

    def parseKoikatsuSunshine(self,cardData:bytes)->dict:
        pos = cardData.find(self.pngEndChunck) + 8
        cardData = cardData[pos:]
        pos = cardData.find(self.pngEndChunck) + 8
        size = int.from_bytes(cardData[pos:pos+4], byteorder="little")
        lstInfoSize = int.from_bytes(cardData[pos:pos+4], byteorder="little") 
        pos += 4
        lstInfo:dict = msgpack.loads(cardData[pos:pos+lstInfoSize])
        pos += lstInfoSize
        dataSize = int.from_bytes(cardData[pos:pos+8], byteorder="little")
        dataPos = pos + 8
        data = cardData[dataPos:dataPos+dataSize]

        d = {}
        _pos = 0
        for item in lstInfo["lstInfo"]:
            name  = item["name"]
            if name == "KKEx":
                continue
            # elif name in ["Custom","Coordinate"]:
            if name == "Custom":
                _pos =item["pos"] if name =="Custom" else item["pos"]+4
                size = item["size"]if name =="Custom" else item["pos"]-4
                _data = data[_pos:_pos+size]
                _pos_ = 0
                l = []
                while _pos_ < size:
                    _size = int.from_bytes(_data[_pos_:_pos_+4], byteorder="little")
                    _pos_ += 4
                    _data_ = msgpack.unpackb(_data[_pos_:_pos_+_size], strict_map_key=False)
                    _pos_ += _size
                    l.append(_data_)
                _data = l
                
            # else:
            #     _pos = item["pos"]
            #     size = item["size"]
            #     _data = msgpack.unpackb(data[_pos:_pos+size], strict_map_key=False)
            d[name] = _data
            # _pos += size



        # d.update(lstInfo)
        return d

    def convertToJson(self,savePath):
        if self.data is None:
            self.parse(self.filePath)

        with open(savePath, 'w') as f:
            json.dump(self.data["Custom"], f)
            
            
    def get_face_data_dict(self,)->dict:
        if self.data is None:
            self.parse(self.filePath)

        data = None
        if self.data is not None:
            if "Custom" in self.data.keys():
                data =  self.data["Custom"]
            else:
                print(f"no found key 'Custom', skip img {self.filePath}")
                return None
        else:
            return None

        face_data_dict = None

        for item in data:
            if "shapeValueFace" in item.keys():
                face_data_dict = item
                break
        
        return face_data_dict
    
    def get_headId(self)->int:
        return int(self.data["Custom"][0]["headId"])

    def convertToLabel(self,):
        face_data_dict = self.get_face_data_dict()
        assert face_data_dict is not None

        face_data = []

        face_data.extend(face_data_dict["shapeValueFace"])
        # face_data.extend(face_data_dict["eyebrowColor"])
        # makeup = face_data_dict["baseMakeup"] if "baseMakeup" in face_data_dict.keys() else face_data_dict["makeup"]
        # face_data.extend(makeup["lipColor"])
        # face_data.extend(makeup["eyeshadowColor"])
        # face_data.extend(makeup["cheekColor"])
        return face_data

    
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname



def generate_labels():
    bar = tqdm(findAllFile("dataset/v0_1000/train/images"))
    labels = {}
    for imagePath in bar:
        characterCard = CharacterCard(imagePath)
        imageName = imagePath.split(os.sep)[-1].split(".")[:-1]
        if len(imageName)>1:
            imageName=".".join(imageName)
        else:
            imageName = imageName[0]
        # savePath = os.path.join("./dataset_json",imageName+".json")
        # characterCard.convertToJson(savePath)
        face_data=characterCard.convertToLabel()
        if face_data is not None:
            dim = len(face_data)
            labels[imageName]=face_data

    with open("./dataset/v0_1000/train/labels.json","w",encoding="utf-8") as f:
        json.dump(labels,f,ensure_ascii=False,indent=4)


    bar = tqdm(findAllFile("dataset/v0_1000/val/images"))
    labels = {}
    dim = 0
    for imagePath in bar:
        characterCard = CharacterCard(imagePath)
        imageName = imagePath.split(os.sep)[-1].split(".")[:-1]
        if len(imageName)>1:
            imageName=".".join(imageName)
        else:
            imageName = imageName[0]
        # savePath = os.path.join("./dataset_json",imageName+".json")
        # characterCard.convertToJson(savePath)
        face_data=characterCard.convertToLabel()
        if face_data is not None:
            dim = len(face_data)
            labels[imageName]=face_data

    with open("./dataset/v0_1000/val/labels.json","w",encoding="utf-8") as f:
        json.dump(labels,f,ensure_ascii=False,indent=4)


    print(f"dim: {dim}")
    
def generate_labels_clean():
    head = [0,1,2]
    bar = tqdm(findAllFile(f"dataset/bak/train/images"))
    labels = {}
    train_counter = np.zeros(11,dtype=np.uint32)
    for imagePath in bar:
        characterCard = CharacterCard(imagePath)
        imageName = imagePath.split(os.sep)[-1].split(".")[:-1]
        if len(imageName)>1:
            imageName=".".join(imageName)
        else:
            imageName = imageName[0]
        # savePath = os.path.join("./dataset_json",imageName+".json")
        # characterCard.convertToJson(savePath)
        headId = characterCard.get_headId()
        if headId<10:
            train_counter[headId] += 1
        else:
            train_counter[10] += 1
        
        if headId not in head:
            os.remove(imagePath)
            print(f"image {imageName} is removed! | headId: {headId}")
            continue
        
        face_data=characterCard.convertToLabel()
        if face_data is not None:
            dim = len(face_data)
            labels[imageName]=face_data

    with open(f"./dataset/bak/train/labels.json","w",encoding="utf-8") as f:
        json.dump(labels,f,ensure_ascii=False,indent=4)


    bar = tqdm(findAllFile(f"dataset/bak/val/images"))
    labels = {}
    val_counter= np.zeros(11,dtype=np.uint8)
    dim = 0
    for imagePath in bar:
        characterCard = CharacterCard(imagePath)
        imageName = imagePath.split(os.sep)[-1].split(".")[:-1]
        if len(imageName)>1:
            imageName=".".join(imageName)
        else:
            imageName = imageName[0]
        # savePath = os.path.join("./dataset_json",imageName+".json")
        # characterCard.convertToJson(savePath)
        if headId<10:
            val_counter[headId] += 1
        else:
            val_counter[10] += 1
        if headId not in head:
            os.remove(imagePath)
            print(f"image {imageName} is removed! | headId: {headId}")
            continue
        face_data=characterCard.convertToLabel()
        if face_data is not None:
            dim = len(face_data)
            labels[imageName]=face_data

    with open(f"./dataset/bak/val/labels.json","w",encoding="utf-8") as f:
        json.dump(labels,f,ensure_ascii=False,indent=4)

    print(f"train head: {train_counter}\nval head: {val_counter}")



if __name__ == "__main__":
    characterCard = CharacterCard("test/test.png")

    # print(characterCard.data)

    data=characterCard.convertToJson("test/test.json")

    print(data)
    # generate_labels()
    # generate_labels_clean()

        
    # bar = tqdm(findAllFile("dataset/new_images"))
    # labels = {}
    # dim = 0
    # for imagePath in bar:
    #     characterCard = CharacterCard(imagePath)
    #     imageName = imagePath.split(os.sep)[-1].split(".")[0]
    #     # savePath = os.path.join("./dataset_json",imageName+".json")
    #     # characterCard.convertToJson(savePath)
    #     face_data=characterCard.convertToLabel()
    #     if face_data is not None:
    #         dim = len(face_data)
    #         labels[imageName]=face_data

    # with open("labels.json","w",encoding="utf-8") as f:
    #     json.dump(labels,f,ensure_ascii=False,indent=4)