import os
import re
from turtle import pos
import msgpack
import json

import numpy as np

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
        # face_data.append(face_data_dict["headId"]/255.0)
        # face_data.append(face_data_dict["skinId"]/255.0)
        # face_data.append(face_data_dict["detailId"]/255.0)
        # face_data.append(face_data_dict["detailPower"])
        # face_data.append(face_data_dict["eyebrowId"]/255.0)
        face_data.extend(face_data_dict["eyebrowColor"])
        # face_data.append(face_data_dict["noseId"]/255.0)
        # pupli = face_data_dict["pupil"][0]
        # face_data.append(pupli["id"]/255.0)
        # face_data.extend(pupli["baseColor"])
        # face_data.extend(pupli["subColor"])
        # face_data.append(pupli["gradMaskId"]/255.0)
        # face_data.append(pupli["gradBlend"])
        # face_data.append(pupli["gradOffsetY"])
        # face_data.append(pupli["gradScale"])
        # face_data.append(face_data_dict["pupilWidth"])
        # face_data.append(face_data_dict["pupilHeight"])
        # face_data.append(face_data_dict["pupilX"])
        # face_data.append(face_data_dict["pupilY"])
        # face_data.append(face_data_dict["hlUpId"]/255.0)
        # face_data.extend(face_data_dict["hlUpColor"])
        # face_data.append(face_data_dict["hlDownId"]/255.0)
        # face_data.extend(face_data_dict["hlDownColor"])        
        # face_data.append(face_data_dict["eyelineUpId"]/255.0)
        # face_data.append(face_data_dict["eyelineDownId"]/255.0)
        # face_data.append(face_data_dict["eyelineUpId"]/255.0)
        # face_data.extend(face_data_dict["eyelineColor"])
        # face_data.append(face_data_dict["moleId"]/255.0)
        # face_data.extend(face_data_dict["moleColor"])
        # face_data.extend(face_data_dict["moleLayout"])
        makeup = face_data_dict["baseMakeup"] if "baseMakeup" in face_data_dict.keys() else face_data_dict["makeup"]
        # face_data.append(makeup["lipId"]/255.0)
        face_data.extend(makeup["lipColor"])
        # face_data.append(face_data_dict["lipLineId"]/255.0)
        # face_data.extend(face_data_dict["lipLineColor"])
        # face_data.append(face_data_dict["lipGlossPower"])
        # face_data.append(makeup["eyeshadowId"]/255.0)
        face_data.extend(makeup["eyeshadowColor"])
        # face_data.append(makeup["cheekId"]/255.0)
        face_data.extend(makeup["cheekColor"])
        # face_data.append(face_data_dict["cheekGlossPower"])




        # with open(savePath, 'w') as f:
        #     json.dump(face_data, f)

        return face_data
    
    def set_face_data(self,vector:np.ndarray)->None:
        data = self.data.copy()

        index = 0
        for item in data["Custom"]:
            if "shapeValueFace" in item.keys():
                break
            index+=1
            
        assert index<=len(data["Custom"])
        data["Custom"][index]["shapeValueFace"]=vector[:59]
        data["Custom"][index]["eyebrowColor"]=vector[59:63]
        key = "baseMakeup" if "baseMakeup" in  data["Custom"][index].keys() else "makeup"
        data["Custom"][index][key]["lipColor"] = vector[63:67]
        data["Custom"][index][key]["eyeshadowColor"] = vector[67:71]
        data["Custom"][index][key]["cheekColor"] = vector[71:75]
        
        self.data = data
        