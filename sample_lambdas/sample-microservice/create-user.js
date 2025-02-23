const userController = require('./controllers/userController');

const createUser = async (event) => {
  try {
    const body = JSON.parse(event.body);
    const user = await userController.createUser(body);
    return {
      statusCode: 201,
      body: JSON.stringify(user),
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};

module.exports = {
  createUser,
};
