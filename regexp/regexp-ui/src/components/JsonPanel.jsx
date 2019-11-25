import React from 'react';
import { Card, CardHeader, CardBody } from 'reactstrap';
import AceEditor from 'react-ace';
import 'brace/theme/github';
import 'brace/mode/json';

export default class JsonPanel extends React.Component {

    // componentDidMount() {
    //     const jsonMode = new SALMode();
    //     this.refs.aceEditor.editor.getSession().setMode(salMode);
    // }

    render() {
        let self = this;
        return (
            <div className="mb-sm-4">
                <Card>
                    <CardHeader>
                        {self.props.title}
                    </CardHeader>
                    <CardBody className="pl-0 pr-0 pb-0 pt-0">
                        <AceEditor
                            mode='json'
                            name={self.props.name}
                            style={{width: '100%'}}
                            fontSize={14}
                            theme="github"
                            showPrintMargin={false}
                            value={self.props.data ? JSON.stringify(self.props.data, null, '\t') : ''}
                            readOnly={true}
                        />
                    </CardBody>
                </Card>
            </div>
        );
    }

}
