import express from "express";
import * as http from "http";

var bodyParser = require('body-parser');
var calOdds = require("../build/Release/calOdds.node");

const app: express.Application = express();
const server: http.Server = http.createServer(app);
const port = 3000;

let shoesMap = new Map();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

function addUsedCardToShoeByShoeId(shoeId:number, newUsedCards:Array<number>) {
  let currentShoeUsedCards = shoesMap.get(shoeId);
  if (currentShoeUsedCards) {
    shoesMap.set(shoeId, currentShoeUsedCards.concat(newUsedCards));
  }else {
    shoesMap.set(shoeId, newUsedCards);
  }
}

app.post('/api/calculateOdds', function(req, res) {
  var shoeDetailJsonStr = req.body.shoeDetail;
  var shoeDetailObj = JSON.parse(shoeDetailJsonStr);
  addUsedCardToShoeByShoeId(shoeDetailObj.shoeId, shoeDetailObj.usedCards);
  console.log(shoesMap);
  
  //TODO: will do calculation job on probability logic and return single interger.
  res.status(200).send('json received: '+ shoeDetailJsonStr);
});

app.get("/", (req: express.Request, res: express.Response) => {
  res
    .status(200)
    .send("testing");
});
server.listen(port, () => {
  console.log(`Server running at port ${port}`);
});
