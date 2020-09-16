import express from "express";
import * as http from "http";

var bodyParser = require('body-parser');
var calOdds = require("../build/Release/calOdds.node");

const app: express.Application = express();
const server: http.Server = http.createServer(app);
const port = 3000;

const CARDID_SHOULD_BE_BETWEEN_ZERO_TO_FIFTY_ONE:string = "cardId should be between 0 and 51";
const CARD_VAL_NOT_ALLOWED:string = "card value occurred more times than is physically possible";
const UNKNOWN_ERROR:string = "unknown error";

let shoesMap = new Map();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

//check card id range from 0 - 51
const regex = RegExp('^[1-4]?[0-9]$|^[5][0-1]$');
function checkIfNewUsedCardsValid(shoeId:number, newUsedCards:Array<number>) {
  //Check if any invalid card id is passed, should be 0 to 51
  if (!newUsedCards.every(function(e){
    return regex.test(e.toString());
  })
  ) {
  return -1;
  }
}

//add the newly used card to specific shoe by shoe id
function addUsedCardToShoeByShoeId(shoeId:number, newUsedCards:Array<number>) {
  //Set the shoe Id and used cards into map for future use
  let currentShoeUsedCards = shoesMap.get(shoeId);
  if (currentShoeUsedCards) {
    shoesMap.set(shoeId, currentShoeUsedCards.concat(newUsedCards));
  }else {
    shoesMap.set(shoeId, newUsedCards);
  }
}

//build result json method
function buildResultJson(odds:number, shoeId:number, status:boolean, message:string) {
  return {
    odds: odds,
    shoeId: shoeId,
    status: status,
    message: message
  };
}

app.post('/api/calculateOdds', function(req, res) {
  let shoeDetailJsonStr = req.body.shoeDetail;
  let shoeDetailObj = JSON.parse(shoeDetailJsonStr);
  let shoeId = shoeDetailObj.shoeId;
  let newUsedCards = shoeDetailObj.usedCards;

  //Do not add the new used card to map directly as error may appear and added trash data
  let odds = calOdds.calOdds(shoesMap.get(shoeId)?shoesMap.get(shoeId).concat(newUsedCards):newUsedCards);
  
  if (odds >= 0) {
    //Only add the new used card here when it is working properly
    addUsedCardToShoeByShoeId(shoeDetailObj.shoeId, shoeDetailObj.usedCards);
    res.status(200).send(JSON.stringify(buildResultJson(odds, shoeDetailObj.shoeId, true, "")));
  }else {
    if (odds == -1) {
      res.status(200).send(JSON.stringify(buildResultJson(-1, -1, false, CARDID_SHOULD_BE_BETWEEN_ZERO_TO_FIFTY_ONE)));
    }else if (odds == -2) {
      res.status(200).send(JSON.stringify(buildResultJson(-2, -2, false, CARD_VAL_NOT_ALLOWED)));
    }else {
      res.status(200).send(JSON.stringify(buildResultJson(-3, -3, false, UNKNOWN_ERROR)));
    }
  }
});

app.get("/", (req: express.Request, res: express.Response) => {
  res
    .status(200)
    .send("testing");
});
server.listen(port, () => {
  console.log(`Server running at port ${port}`);
});
