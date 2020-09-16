import express from "express";
import * as http from "http";

var bodyParser = require('body-parser');
var calOdds = require("../build/Release/calOdds.node");

const app: express.Application = express();
const server: http.Server = http.createServer(app);
const port = 3000;

//set of return messages
const CARDID_SHOULD_BE_BETWEEN_ZERO_TO_FIFTY_ONE:string = "cardId should be between 0 and 51";
const CARD_VAL_NOT_ALLOWED:string = "card value occurred more times than is physically possible";
const JSON_PARSE_ERR:string = "json format is not correct";
const UNKNOWN_ERROR:string = "unknown error";
const RESET_SHOE:string = "shoe reset";

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

//map for storing used cards of shoes
let shoesMap = new Map();

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

//send response after calculation
function sendResponse(res:any, httpStatus:number, odds:number, shoeId:number, status:boolean, message:string) {
  res.status(httpStatus).send(JSON.stringify(buildResultJson(odds, shoeId, status, message)));
}


//reset the shoe
function resetShoe(shoeId:number) {
  shoesMap.set(shoeId, null);
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

//API for calculating odds for shoes
app.post('/api/calculateOdds', function(req, res) {
  let shoeDetailJsonStr = req.body.shoeDetail;
  let shoeDetailObj;

  //validate json format
  try {
    shoeDetailObj = JSON.parse(shoeDetailJsonStr);
  }catch (e) {
    sendResponse(res, 400, -4, -4, false, JSON_PARSE_ERR);
    return;
  }
  let shoeId:number = shoeDetailObj.shoeId;
  let newUsedCards:Array<number> = shoeDetailObj.usedCards;
  let clearShoe:boolean = shoeDetailObj.clearShoe;

  if (!clearShoe) {
    //Do not add the new used card to map directly as error may appear and added trash data
    let odds = calOdds.calOdds(shoesMap.get(shoeId)?shoesMap.get(shoeId).concat(newUsedCards):newUsedCards);
    
    if (odds >= 0) {
      //Only add the new used card here when it is working properly
      addUsedCardToShoeByShoeId(shoeDetailObj.shoeId, shoeDetailObj.usedCards);
      sendResponse(res, 200, odds, shoeDetailObj.shoeId, true, "");
    }else {
      if (odds == -1) {
        sendResponse(res, 400, -1, -1, false, CARDID_SHOULD_BE_BETWEEN_ZERO_TO_FIFTY_ONE);
      }else if (odds == -2) {
        sendResponse(res, 400, -2, -2, false, CARD_VAL_NOT_ALLOWED);
      }else {
        sendResponse(res, 400, -3, -3, false, UNKNOWN_ERROR);
      }
    }
  }else {
    resetShoe(shoeId);
    sendResponse(res, 200, -999, -999, true, RESET_SHOE);
  }
  
});

server.listen(port, () => {
  console.log(`Server running at port ${port}`);
});
