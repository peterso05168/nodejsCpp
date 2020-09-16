import express from "express";
import * as http from "http";

var bodyParser = require('body-parser');
var greet = require("../build/Release/greet.node");

const app: express.Application = express();
const server: http.Server = http.createServer(app);
const port = 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

app.post('/api/calculateOdds', function(req, res) {
  var usedCardsJsonStr = req.body.usedCards;
  var usedCardsArray = JSON.parse(usedCardsJsonStr);
  console.log(usedCardsArray);
  //TODO: will do calculation job on probability logic and return single interger.
  res.status(200).send('Array received: ' + usedCardsJsonStr);
});

app.get("/", (req: express.Request, res: express.Response) => {
  res
    .status(200)
    .send(`Server running at port ${port}` + greet.greetHello("test"));
});
server.listen(port, () => {
  console.log(`Server running at port ${port}`);
});
